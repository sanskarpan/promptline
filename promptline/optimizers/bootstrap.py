"""Bootstrap few-shot optimizers.

``BootstrapFewShot`` collects passing demonstrations from the training set and
attaches them as few-shot demos to the seed candidate.

``BootstrapRandomSearch`` samples multiple subsets of those demonstrations and
picks the one that scores highest on a held-out validation split.
"""

from __future__ import annotations

import inspect
import random
from collections.abc import Callable

from promptline.core.program import PromptProgram
from promptline.core.types import Candidate, Demo, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, Metric
from promptline.optimizers.base import Optimizer, OptimizeResult, RunEvent

# ---------------------------------------------------------------------------
# Shared demo collection
# ---------------------------------------------------------------------------


async def collect_demo_pool(
    program: PromptProgram,
    candidate: Candidate,
    examples: list[Example],
    metric: Metric,
    budget: Budget,
    harness: EvalHarness,
    threshold: float,
    max_per_module: int,
) -> tuple[dict[str, list[Demo]], int, int]:
    """Rejection-sample passing examples into per-module demo pools.

    Runs *candidate* over *examples* (budget-charged, one rollout each) and,
    for every example whose metric score reaches *threshold*, records a
    :class:`Demo` per module from the prediction traces.  Stops when every
    module has *max_per_module* demos, when the budget is exhausted, or when
    the examples run out.

    Returns ``(module_demos, passed, total)`` where *passed* / *total* count
    examples that reached the threshold vs. examples attempted.

    .. note::
        Multi-module caveat: only the first module's inputs are reliably
        known (they are ``example.inputs``); later modules' inputs come from
        prior-module outputs which are not tracked separately here — we fall
        back to ``example.inputs`` for all modules, which may be incomplete.
    """
    module_demos: dict[str, list[Demo]] = {m.name: [] for m in program.modules}
    passed = 0
    total = 0

    for example in examples:
        # Stop if budget is already exhausted.
        if budget.exhausted:
            break

        # Stop once every module has enough demos.
        all_full = all(len(module_demos[m.name]) >= max_per_module for m in program.modules)
        if all_full:
            break

        # Atomically reserve a rollout slot.
        reserved = await budget.try_reserve(rollouts=1)
        if not reserved:
            break

        total += 1

        try:
            prediction = await program.run(example, candidate, harness.client, harness.cfg)
            budget.add_cost(prediction.cost_usd)
        except Exception:
            continue

        if prediction.failed:
            continue

        raw = metric(example, prediction)
        if inspect.isawaitable(raw):
            raw = await raw

        if raw.score >= threshold:
            passed += 1
            for trace in prediction.traces:
                mod_name = trace.module
                if mod_name not in module_demos:
                    continue
                if len(module_demos[mod_name]) >= max_per_module:
                    continue
                if trace.parsed is None:
                    continue
                module_demos[mod_name].append(
                    Demo(inputs=dict(example.inputs), outputs=trace.parsed)
                )

    return module_demos, passed, total


class BootstrapFewShot(Optimizer):
    """Collect passing training examples as few-shot demonstrations.

    Parameters
    ----------
    max_demos:
        Maximum number of demonstrations to attach per module.
    threshold:
        Minimum metric score (inclusive) for an example to qualify as a demo.
        The default (0.7) suits continuous metrics like the LLM judge, which
        produces scores in [0, 1]; binary exact-match scores (1.0/0.0) clear
        it too.  Pass 1.0 to accept only perfect scores.
    rng_seed:
        Seed used to shuffle the training set before collection.
    """

    name = "bootstrap"

    def __init__(
        self,
        max_demos: int = 4,
        threshold: float = 0.7,
        rng_seed: int = 0,
    ) -> None:
        self.max_demos = max_demos
        self.threshold = threshold
        self.rng_seed = rng_seed

    async def optimize(
        self,
        program: PromptProgram,
        seed: Candidate,
        trainset: list[Example],
        metric: Metric,
        budget: Budget,
        harness: EvalHarness,
        emit: Callable[[RunEvent], None] = lambda e: None,
    ) -> OptimizeResult:
        events: list[RunEvent] = []

        def _emit(event: RunEvent) -> None:
            events.append(event)
            emit(event)

        _emit(RunEvent.now("run_started", optimizer=self.name))

        rng = random.Random(self.rng_seed)
        shuffled = list(trainset)
        rng.shuffle(shuffled)

        # Collect per-module demos via shared rejection sampling.
        module_demos, passed, total = await collect_demo_pool(
            program,
            seed,
            shuffled,
            metric,
            budget,
            harness,
            threshold=self.threshold,
            max_per_module=self.max_demos,
        )

        # Build the best candidate with collected demos.
        new_modules: dict[str, ModuleState] = {
            mod_name: ModuleState(
                instruction=seed.modules[mod_name].instruction,
                demos=module_demos.get(mod_name, []),
            )
            for mod_name in seed.modules
        }
        best = seed.child(modules=new_modules, optimizer=self.name)

        # One budget_tick after the collection loop so consumers see burn-down.
        _emit(
            RunEvent.now(
                "budget_tick",
                rollouts_used=budget.rollouts_used,
                cost_used=budget.cost_used,
                max_rollouts=budget.max_rollouts,
                max_cost_usd=budget.max_cost_usd,
            )
        )

        _emit(
            RunEvent.now(
                "candidate_proposed",
                candidate_id=best.id,
                parents=[seed.id],
                demos_count={m: len(d) for m, d in module_demos.items()},
            )
        )

        # Seed pass-rate is recorded under seed.id.
        seed_score = passed / total if total > 0 else 0.0
        scores: dict[str, float] = {seed.id: seed_score}

        # Evaluate the augmented candidate on the collection set if budget allows,
        # and record its true mean score under best.id.  If budget is already
        # exhausted, omit best.id from scores rather than recording a stale value.
        if not budget.exhausted:
            report = await harness.evaluate(program, best, shuffled, metric, budget)
            scores[best.id] = report.mean_score

        finished_payload: dict = {"optimizer": self.name, "best_id": best.id}
        if best.id in scores:
            finished_payload["best_score"] = scores[best.id]
        _emit(RunEvent.now("run_finished", **finished_payload))

        return OptimizeResult(
            best=best,
            candidates=[seed, best],
            scores=scores,
            events_count=len(events),
        )


class BootstrapRandomSearch(Optimizer):
    """Bootstrap a demo pool, then search over random subsets.

    Parameters
    ----------
    n_subsets:
        Number of random demo subsets to evaluate.
    subset_size:
        Number of demos per subset (may be smaller if pool is short).
    threshold:
        Minimum score for an example to enter the demo pool.  The default
        (0.7) suits continuous metrics like the LLM judge ([0, 1] scores);
        binary 1.0/0.0 metrics clear it too.
    val_fraction:
        Fraction of the training set reserved for validation (seeded split).
    rng_seed:
        Seed for all random operations (shuffle, split, subset sampling).
    """

    name = "bootstrap-rs"

    def __init__(
        self,
        n_subsets: int = 8,
        subset_size: int = 4,
        threshold: float = 0.7,
        val_fraction: float = 0.3,
        rng_seed: int = 0,
    ) -> None:
        self.n_subsets = n_subsets
        self.subset_size = subset_size
        self.threshold = threshold
        self.val_fraction = val_fraction
        self.rng_seed = rng_seed

    async def optimize(
        self,
        program: PromptProgram,
        seed: Candidate,
        trainset: list[Example],
        metric: Metric,
        budget: Budget,
        harness: EvalHarness,
        emit: Callable[[RunEvent], None] = lambda e: None,
    ) -> OptimizeResult:
        events: list[RunEvent] = []

        def _emit(event: RunEvent) -> None:
            events.append(event)
            emit(event)

        _emit(RunEvent.now("run_started", optimizer=self.name))

        rng = random.Random(self.rng_seed)
        shuffled = list(trainset)
        rng.shuffle(shuffled)

        # Split into pool-source and validation sets.
        n_val = max(1, int(len(shuffled) * self.val_fraction))
        val_set = shuffled[:n_val]
        pool_source = shuffled[n_val:]

        # Bootstrap a demo POOL from pool_source.
        max_pool = self.n_subsets * self.subset_size
        pool: list[Demo] = []
        # Track which module to collect from (first module for simplicity).
        first_mod = program.modules[0].name if program.modules else ""

        for example in pool_source:
            if len(pool) >= max_pool:
                break
            if budget.exhausted:
                break

            reserved = await budget.try_reserve(rollouts=1)
            if not reserved:
                break

            try:
                prediction = await program.run(example, seed, harness.client, harness.cfg)
                budget.add_cost(prediction.cost_usd)
            except Exception:
                continue

            if prediction.failed:
                continue

            raw = metric(example, prediction)
            if inspect.isawaitable(raw):
                raw = await raw

            if raw.score >= self.threshold:
                # Collect from first module trace.
                for trace in prediction.traces:
                    if trace.module == first_mod and trace.parsed is not None:
                        pool.append(Demo(inputs=dict(example.inputs), outputs=trace.parsed))
                        break

        # Sample n_subsets random subsets from the pool and evaluate each.
        all_candidates: list[Candidate] = [seed]
        all_scores: dict[str, float] = {}
        best_candidate = seed
        best_score = -1.0

        for subset_idx in range(self.n_subsets):
            if budget.exhausted:
                break

            _emit(
                RunEvent.now(
                    "budget_tick",
                    subset_idx=subset_idx,
                    rollouts_used=budget.rollouts_used,
                    cost_used=budget.cost_used,
                    max_rollouts=budget.max_rollouts,
                    max_cost_usd=budget.max_cost_usd,
                )
            )

            if not pool:
                # Empty pool — evaluate seed as fallback.
                chosen: list[Demo] = []
            elif len(pool) <= self.subset_size:
                chosen = list(pool)
            else:
                chosen = rng.sample(pool, self.subset_size)

            # Build a candidate with this subset of demos.
            new_modules: dict[str, ModuleState] = {}
            for mod_name, state in seed.modules.items():
                if mod_name == first_mod:
                    new_modules[mod_name] = ModuleState(
                        instruction=state.instruction,
                        demos=chosen,
                    )
                else:
                    new_modules[mod_name] = ModuleState(
                        instruction=state.instruction,
                        demos=[],
                    )
            candidate = seed.child(modules=new_modules, optimizer=self.name)
            all_candidates.append(candidate)

            _emit(
                RunEvent.now(
                    "candidate_proposed",
                    candidate_id=candidate.id,
                    parents=[seed.id],
                    subset_idx=subset_idx,
                    demos_count=len(chosen),
                )
            )

            # Evaluate on validation set.
            report = await harness.evaluate(program, candidate, val_set, metric, budget)
            score = report.mean_score
            all_scores[candidate.id] = score

            _emit(
                RunEvent.now(
                    "full_eval",
                    candidate_id=candidate.id,
                    subset_idx=subset_idx,
                    mean_score=score,
                )
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate

        # Guarantee the returned best always has a score entry so callers
        # (including the CLI) can safely look it up without a missing-key crash.
        if best_candidate.id not in all_scores:
            all_scores[best_candidate.id] = 0.0

        _emit(
            RunEvent.now(
                "run_finished",
                optimizer=self.name,
                best_id=best_candidate.id,
                best_score=all_scores[best_candidate.id],
            )
        )

        return OptimizeResult(
            best=best_candidate,
            candidates=all_candidates,
            scores=all_scores,
            events_count=len(events),
        )
