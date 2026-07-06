"""GEPA engine — Genetic-Pareto reflective prompt evolution (arXiv:2507.19457).

Algorithm 1: maintain a pool of candidates scored per-instance on a held-out
``D_pareto`` split.  Each iteration samples a candidate from the per-instance
Pareto frontier (Algorithm 2), mutates one module's instruction via an LLM
reflection over minibatch traces and feedback, and accepts the child only on a
strict minibatch improvement.  Periodically, two frontier candidates from
different lineages are recombined with a system-aware merge (Appendix F).
"""

from __future__ import annotations

import inspect
import random
from collections.abc import Callable
from pathlib import Path

from promptline.core.llm import LLMCall, Message
from promptline.core.program import REPAIR_PROMPT, Prediction, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, Metric, MetricResult
from promptline.optimizers.base import (
    Optimizer,
    OptimizeResult,
    RunEvent,
    RunRecorder,
)
from promptline.optimizers.gepa.merge import (
    common_ancestor,
    is_related,
    merge_candidates,
)
from promptline.optimizers.gepa.pareto import frontier_candidates, pareto_sample
from promptline.optimizers.gepa.reflect import (
    ReflectionExample,
    build_reflection_prompt,
    parse_new_instruction,
)
from promptline.optimizers.gepa.state import GepaState

#: One evaluated minibatch example: (example, prediction, metric result).
_MinibatchItem = tuple[Example, Prediction, MetricResult]


class GEPA(Optimizer):
    """Genetic-Pareto prompt optimizer.

    Parameters
    ----------
    minibatch_size:
        Number of ``D_feedback`` examples used for each reflect/accept step.
    n_pareto:
        Maximum size of the ``D_pareto`` split (capped at half the trainset).
    use_merge, max_merges, merge_every:
        Enable system-aware merges, cap their number, and attempt one every
        *merge_every* accepted candidates.
    max_iterations:
        Safety cap on loop iterations (budget is the primary stop).
    rng_seed:
        Seed for the split, minibatch sampling and Pareto sampling.
    run_dir:
        Directory for checkpoints (written after every acceptance).
    resume_from:
        Resume from the checkpoint in this directory (also used as
        ``run_dir`` when the latter is unset).

    Notes
    -----
    **RNG state is not checkpointed.**  Resumed runs re-seed ``random.Random``
    from ``rng_seed`` and therefore replay a *different* sampling sequence than
    an uninterrupted run would have.  Candidate ids are preserved across resume
    (restored from the checkpoint), but the minibatch and Pareto-sampling
    sequences diverge from what the uninterrupted run would have produced.
    """

    name = "gepa"

    def __init__(
        self,
        minibatch_size: int = 3,
        n_pareto: int = 32,
        use_merge: bool = True,
        max_merges: int = 5,
        merge_every: int = 7,
        max_iterations: int = 200,
        rng_seed: int = 0,
        run_dir: Path | None = None,
        resume_from: Path | None = None,
    ) -> None:
        self.minibatch_size = minibatch_size
        self.n_pareto = n_pareto
        self.use_merge = use_merge
        self.max_merges = max_merges
        self.merge_every = merge_every
        self.max_iterations = max_iterations
        self.rng_seed = rng_seed
        self.resume_from = resume_from
        if run_dir is None:
            run_dir = resume_from
        self._recorder = RunRecorder(run_dir) if run_dir is not None else None

    # ------------------------------------------------------------------
    # Data split
    # ------------------------------------------------------------------

    def _split(self, trainset: list[Example]) -> tuple[list[Example], list[Example]]:
        """Seeded split of *trainset* into (D_feedback, D_pareto).

        Deterministic for a given ``rng_seed`` so a resumed run sees the
        identical split (score vectors stay index-aligned).
        """
        rng = random.Random(self.rng_seed)
        indices = list(range(len(trainset)))
        rng.shuffle(indices)
        n_p = min(self.n_pareto, max(1, len(trainset) // 2))
        pareto_idx = sorted(indices[:n_p])
        feedback_idx = sorted(indices[n_p:])
        d_pareto = [trainset[i] for i in pareto_idx]
        # Degenerate trainsets (1 example): reuse D_pareto for feedback.
        d_feedback = [trainset[i] for i in feedback_idx] or list(d_pareto)
        return d_feedback, d_pareto

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    async def _run_minibatch(
        self,
        program: PromptProgram,
        candidate: Candidate,
        batch: list[Example],
        metric: Metric,
        budget: Budget,
        harness: EvalHarness,
    ) -> list[_MinibatchItem] | None:
        """Run *candidate* on *batch* keeping traces; ``None`` on budget wall.

        The harness discards traces, and reflection needs them, so minibatches
        run via direct ``program.run`` calls — still reserving one rollout per
        example against *budget*.
        """
        results: list[_MinibatchItem] = []
        for example in batch:
            if not await budget.try_reserve(rollouts=1):
                return None
            prediction = await program.run(example, candidate, harness.client, harness.cfg)
            budget.add_cost(prediction.cost_usd)
            if prediction.failed:
                mr = MetricResult(score=0.0, feedback=prediction.failure_reason)
            else:
                raw = metric(example, prediction)
                if inspect.isawaitable(raw):
                    raw = await raw
                mr = raw
            results.append((example, prediction, mr))
        return results

    @staticmethod
    def _mean(items: list[_MinibatchItem]) -> float:
        if not items:
            return 0.0
        return sum(mr.score for _, _, mr in items) / len(items)

    async def _full_eval(
        self,
        state: GepaState,
        program: PromptProgram,
        candidate: Candidate,
        d_pareto: list[Example],
        metric: Metric,
        budget: Budget,
        harness: EvalHarness,
        emit: Callable[[RunEvent], None],
    ) -> None:
        """Full-eval *candidate* on ``D_pareto``, record S[candidate] and emit."""
        report = await harness.evaluate(program, candidate, d_pareto, metric, budget)
        vector = [0.0] * len(d_pareto)
        for res in report.per_example:
            vector[res.example_idx] = res.score
        state.add(candidate, vector)
        if report.truncated or report.n < len(d_pareto):
            state.partial.add(candidate.id)
        else:
            state.partial.discard(candidate.id)
        emit(
            RunEvent.now(
                "full_eval",
                candidate_id=candidate.id,
                mean_score=state.mean(candidate.id),
                n=report.n,
                truncated=report.truncated,
            )
        )
        emit(
            RunEvent.now(
                "pareto_updated",
                candidate_id=candidate.id,
                frontier=sorted(frontier_candidates(state.scores)),
                pool_size=len(state.pool),
            )
        )

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    async def _reflect(
        self,
        candidate: Candidate,
        module_name: str,
        minibatch: list[_MinibatchItem],
        budget: Budget,
        harness: EvalHarness,
        iteration: int,
    ) -> str:
        """One reflection LLM call → proposed new instruction for *module_name*."""
        examples: list[ReflectionExample] = []
        for example, prediction, mr in minibatch:
            trace = next(
                (
                    t
                    for t in prediction.traces
                    if t.module == module_name and t.user_prompt != REPAIR_PROMPT
                ),
                None,
            )
            if trace is not None:
                inputs, output = trace.user_prompt, trace.raw_output
            else:
                inputs = "\n".join(f"{k}: {v}" for k, v in example.inputs.items())
                output = "(module did not run)"
            feedback = mr.per_module.get(module_name) or mr.feedback
            examples.append(
                ReflectionExample(inputs=inputs, output=output, score=mr.score, feedback=feedback)
            )

        prompt = build_reflection_prompt(
            module_name, candidate.modules[module_name].instruction, examples
        )
        reflection_model = harness.cfg.reflection_model or harness.cfg.task_model
        call = LLMCall(
            model=reflection_model,
            messages=(Message(role="user", content=prompt),),
            temperature=1.0,
            max_tokens=harness.cfg.max_tokens,
            # Distinct seed per iteration so a caching client cannot collapse
            # successive reflections into one cached response.
            seed=iteration,
        )
        response = await harness.client.complete(call)
        budget.add_cost(response.cost_usd)
        return parse_new_instruction(response.text)

    @staticmethod
    def _mutate(candidate: Candidate, module_name: str, instruction: str) -> Candidate:
        """Child of *candidate* with *module_name*'s instruction replaced."""
        modules: dict[str, ModuleState] = {}
        for name, mod_state in candidate.modules.items():
            if name == module_name:
                modules[name] = ModuleState(instruction=instruction, demos=list(mod_state.demos))
            else:
                modules[name] = mod_state.model_copy(deep=True)
        return candidate.child(modules=modules, optimizer=GEPA.name)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    async def _attempt_merge(
        self,
        state: GepaState,
        program: PromptProgram,
        d_feedback: list[Example],
        d_pareto: list[Example],
        metric: Metric,
        budget: Budget,
        harness: EvalHarness,
        rng: random.Random,
        emit: Callable[[RunEvent], None],
    ) -> bool:
        """Try one system-aware merge; returns True when a child was accepted.

        Acceptance criterion (Appendix F): the merged child's minibatch score must
        be >= the better of its two parents (not strict >).  A tied score is treated
        as a win because system-aware merge reduces per-module variance without
        requiring a net improvement on the shared minibatch.
        """
        frontier = sorted(frontier_candidates(state.scores))
        pairs = [
            (a, b)
            for i, a in enumerate(frontier)
            for b in frontier[i + 1 :]
            if not is_related(a, b, state.pool)
        ]
        rng.shuffle(pairs)

        chosen: tuple[str, str, str] | None = None
        for a, b in pairs:
            ancestor_id = common_ancestor(a, b, state.pool)
            if ancestor_id is not None:
                chosen = (a, b, ancestor_id)
                break
        if chosen is None:
            emit(
                RunEvent.now(
                    "merge_attempted",
                    parents=[],
                    ancestor=None,
                    accepted=False,
                    reason="no mergeable pair",
                )
            )
            return False

        p1_id, p2_id, ancestor_id = chosen
        p1, p2 = state.pool[p1_id], state.pool[p2_id]
        child = merge_candidates(
            p1,
            p2,
            state.pool[ancestor_id],
            state.mean(p1_id),
            state.mean(p2_id),
            rng,
            optimizer=self.name,
        )
        emit(
            RunEvent.now(
                "candidate_proposed",
                candidate_id=child.id,
                kind="merge",
                parents=[p1_id, p2_id],
            )
        )

        # Fresh minibatch; all three candidates scored on the same examples.
        batch = rng.sample(d_feedback, min(self.minibatch_size, len(d_feedback)))
        scores: dict[str, float] = {}
        truncated = False
        for cand in (p1, p2, child):
            report = await harness.evaluate(program, cand, batch, metric, budget)
            truncated = truncated or report.truncated or report.n < len(batch)
            scores[cand.id] = report.mean_score
            emit(
                RunEvent.now(
                    "minibatch_scored",
                    candidate_id=cand.id,
                    mean_score=report.mean_score,
                    kind="merge",
                )
            )

        accepted = not truncated and scores[child.id] >= max(scores[p1_id], scores[p2_id])
        emit(
            RunEvent.now(
                "merge_attempted",
                parents=[p1_id, p2_id],
                ancestor=ancestor_id,
                accepted=accepted,
            )
        )
        if accepted:
            await self._full_eval(state, program, child, d_pareto, metric, budget, harness, emit)
            state.accepted_count += 1
            self._checkpoint(state, budget)
        return accepted

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint(self, state: GepaState, budget: Budget) -> None:
        if self._recorder is None:
            return
        payload = state.to_dict()
        payload["budget_spent"] = {
            "rollouts": budget.rollouts_used,
            "cost_usd": budget.cost_used,
        }
        self._recorder.save_checkpoint(payload)

    def _load_state(self) -> GepaState | None:
        if self.resume_from is None:
            return None
        checkpoint = RunRecorder(self.resume_from).load_checkpoint()
        if not checkpoint.get("candidates"):
            return None
        return GepaState.from_dict(checkpoint)

    # ------------------------------------------------------------------
    # Main loop (Algorithm 1)
    # ------------------------------------------------------------------

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
        events_count = 0

        def _emit(event: RunEvent) -> None:
            nonlocal events_count
            events_count += 1
            if self._recorder is not None:
                self._recorder.emit(event)
            emit(event)

        d_feedback, d_pareto = self._split(trainset)
        rng = random.Random(self.rng_seed)
        module_names = program.module_names

        resumed_state = self._load_state()
        state = resumed_state if resumed_state is not None else GepaState()

        _emit(
            RunEvent.now(
                "run_started",
                optimizer=self.name,
                resumed=resumed_state is not None,
                n_pareto=len(d_pareto),
                n_feedback=len(d_feedback),
            )
        )

        # Step 0 (fresh runs): full-eval the seed on D_pareto.
        if seed.id not in state.pool:
            if resumed_state is not None:
                # On resume the seed id may differ from what was checkpointed.
                # If a pool candidate has identical module content, treat it as
                # the seed (no re-eval, no duplicate).
                matching = next(
                    (cid for cid, cand in state.pool.items() if cand.modules == seed.modules),
                    None,
                )
                if matching is None:
                    raise ValueError("resume pool does not contain the provided seed")
                # else: identical candidate already in pool; skip re-eval.
            else:
                await self._full_eval(
                    state, program, seed, d_pareto, metric, budget, harness, _emit
                )
                self._checkpoint(state, budget)

        # Repair partial Pareto vectors before main loop (resume only).
        if resumed_state is not None:
            for cid in list(state.partial):
                if not budget.exhausted:
                    await self._full_eval(
                        state,
                        program,
                        state.pool[cid],
                        d_pareto,
                        metric,
                        budget,
                        harness,
                        _emit,
                    )

        # ----------------------------------------------------------------
        # Main loop
        # ----------------------------------------------------------------
        while state.iteration < self.max_iterations and not budget.exhausted:
            state.iteration += 1
            _emit(
                RunEvent.now(
                    "budget_tick",
                    rollouts_used=budget.rollouts_used,
                    cost_used=budget.cost_used,
                    max_rollouts=budget.max_rollouts,
                    max_cost_usd=budget.max_cost_usd,
                )
            )

            do_merge = (
                self.use_merge
                and state.merges_done < self.max_merges
                and state.accepts_since_merge >= self.merge_every
                and len(state.pool) >= 2
            )
            if do_merge:
                state.merges_done += 1
                state.accepts_since_merge = 0
                await self._attempt_merge(
                    state,
                    program,
                    d_feedback,
                    d_pareto,
                    metric,
                    budget,
                    harness,
                    rng,
                    _emit,
                )
                continue

            # (a) Pareto-sample a parent from the pool.
            parent_id = pareto_sample(state.scores, rng)
            parent = state.pool[parent_id]

            # (b) Round-robin module selection.
            module_name = module_names[state.module_counter % len(module_names)]
            state.module_counter += 1

            # (c) Seeded minibatch from D_feedback.
            batch = rng.sample(d_feedback, min(self.minibatch_size, len(d_feedback)))

            # (d) Run the parent on the minibatch, keeping traces.
            parent_run = await self._run_minibatch(program, parent, batch, metric, budget, harness)
            if parent_run is None:
                break  # budget wall mid-minibatch
            parent_score = self._mean(parent_run)
            _emit(
                RunEvent.now(
                    "minibatch_scored",
                    candidate_id=parent_id,
                    mean_score=parent_score,
                    iteration=state.iteration,
                    role="parent",
                )
            )

            # (e) Reflect and build the child candidate.
            new_instruction = await self._reflect(
                parent, module_name, parent_run, budget, harness, state.iteration
            )
            child = self._mutate(parent, module_name, new_instruction)
            _emit(
                RunEvent.now(
                    "candidate_proposed",
                    candidate_id=child.id,
                    parents=[parent_id],
                    parent_id=parent_id,  # legacy alias
                    module=module_name,
                    iteration=state.iteration,
                    instruction=new_instruction,
                )
            )

            # (f) Score the child on the same minibatch; strict acceptance.
            child_run = await self._run_minibatch(program, child, batch, metric, budget, harness)
            if child_run is None:
                break
            child_score = self._mean(child_run)
            _emit(
                RunEvent.now(
                    "minibatch_scored",
                    candidate_id=child.id,
                    mean_score=child_score,
                    iteration=state.iteration,
                    role="child",
                    accepted=child_score > parent_score,
                )
            )
            if child_score <= parent_score:
                continue

            # (g) Accepted: full-eval on D_pareto and checkpoint.
            await self._full_eval(state, program, child, d_pareto, metric, budget, harness, _emit)
            state.accepted_count += 1
            state.accepts_since_merge += 1
            self._checkpoint(state, budget)

        # ----------------------------------------------------------------
        # Best candidate by mean D_pareto score.
        # ----------------------------------------------------------------
        best_id = state.best_id()
        _emit(
            RunEvent.now(
                "run_finished",
                optimizer=self.name,
                best_id=best_id,
                best_score=state.mean(best_id),
                pool_size=len(state.pool),
                iterations=state.iteration,
                merges_done=state.merges_done,
            )
        )

        return OptimizeResult(
            best=state.pool[best_id],
            candidates=list(state.pool.values()),
            scores={cid: state.mean(cid) for cid in state.pool},
            events_count=events_count,
        )
