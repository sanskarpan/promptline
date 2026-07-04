"""MIPRO-like optimizer (arXiv:2406.11695).

Three stages:

1. **Demo sets** — bootstrap a pool of passing demonstrations per module, then
   build several candidate demo sets (set 0 is always empty / zero-shot).
2. **Grounded instruction proposal** — summarise the dataset with one LLM call,
   build a programmatic program summary, then propose alternative instructions
   per module grounded in both (plus example demos and a randomised tip).
3. **Bayesian search** — a TPE study over the categorical space
   ``inst_<module> x demo_<module>``, scoring configs on fresh minibatches with
   periodic full evaluations of the most promising configs.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Callable

import optuna

from promptline.core.llm import LLMCall, Message
from promptline.core.program import PromptProgram
from promptline.core.types import Candidate, Demo, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, Metric
from promptline.optimizers.base import Optimizer, OptimizeResult, RunEvent
from promptline.optimizers.bootstrap import collect_demo_pool
from promptline.optimizers.gepa.reflect import parse_new_instruction

# Silence optuna's per-trial INFO chatter once, at import time.
optuna.logging.set_verbosity(optuna.logging.WARNING)

#: Style tips randomly injected into proposal prompts (one per proposal).
TIPS = [
    "Be creative.",
    "Keep it simple and direct.",
    "Be highly descriptive and specific.",
    "Emphasize accuracy — mistakes are high-stakes.",
    "Adopt an expert persona.",
]

#: Config key type: (sorted inst items, sorted demo items).
_ConfigKey = tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _render_example(example: Example) -> str:
    parts = [f"{k}: {v}" for k, v in example.inputs.items()]
    parts += [f"{k} (label): {v}" for k, v in example.labels.items()]
    return "\n".join(parts)


def _build_dataset_summary_prompt(samples: list[Example]) -> str:
    lines: list[str] = ["Here are some examples from a dataset:"]
    for i, ex in enumerate(samples, start=1):
        lines += ["", f"### Example {i}", _render_example(ex)]
    lines += ["", "Summarize the patterns in this dataset in 3-5 bullet points."]
    return "\n".join(lines)


def _build_program_summary(program: PromptProgram, seed: Candidate) -> str:
    lines: list[str] = ["The LLM program has the following modules:"]
    for module in program.modules:
        sig = module.signature
        inputs = ", ".join(f.name for f in sig.inputs)
        outputs = ", ".join(f.name for f in sig.outputs)
        instruction = seed.modules[module.name].instruction
        lines.append(
            f"- {module.name}: inputs=({inputs}) outputs=({outputs}) "
            f'instruction="{instruction}"'
        )
    return "\n".join(lines)


def _build_proposal_prompt(
    dataset_summary: str,
    program_summary: str,
    module_name: str,
    current_instruction: str,
    demos: list[Demo],
    previous_proposals: list[str],
    tip: str,
) -> str:
    lines: list[str] = [
        "You are an expert prompt engineer improving one module of an LLM program.",
        "",
        "Dataset summary:",
        dataset_summary,
        "",
        "Program summary:",
        program_summary,
        "",
        f'You are improving the module "{module_name}".',
        "Its current instruction is:",
        "```",
        current_instruction,
        "```",
    ]
    if demos:
        lines += ["", "Example demonstrations for this module:"]
        for demo in demos[:2]:
            demo_in = "\n".join(f"{k}: {v}" for k, v in demo.inputs.items())
            demo_out = "\n".join(f"{k}: {v}" for k, v in demo.outputs.items())
            lines += ["", "Input:", demo_in, "Output:", demo_out]
    if previous_proposals:
        lines += ["", "Instructions already proposed in this batch (write something different):"]
        lines += [f"- {p}" for p in previous_proposals]
    lines += [
        "",
        f"Tip: {tip}",
        "",
        "Write an improved instruction for this module. "
        "Output ONLY the instruction in a fenced code block.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class MIPRO(Optimizer):
    """MIPRO-like optimizer: bootstrapped demos + grounded instructions + TPE.

    Parameters
    ----------
    n_instruction_candidates:
        Instruction options per module (option 0 is the seed instruction).
    n_demo_sets:
        Demo-set options per module (set 0 is always empty / zero-shot).
    demos_per_set:
        Maximum demos per non-empty demo set.
    n_trials:
        Maximum number of TPE trials (early-stopped by budget).
    minibatch_size:
        Examples per trial minibatch (fresh seeded sample per trial).
    full_eval_steps:
        Every this many completed trials, full-evaluate the best
        not-yet-full-evaluated config on the whole trainset.
    threshold:
        Minimum metric score for an example to enter the demo pool.
    rng_seed:
        Seed for all random operations and the TPE sampler.
    """

    name = "mipro"

    def __init__(
        self,
        n_instruction_candidates: int = 6,
        n_demo_sets: int = 4,
        demos_per_set: int = 4,
        n_trials: int = 30,
        minibatch_size: int = 16,
        full_eval_steps: int = 5,
        threshold: float = 1.0,
        rng_seed: int = 0,
    ) -> None:
        self.n_instruction_candidates = n_instruction_candidates
        self.n_demo_sets = n_demo_sets
        self.demos_per_set = demos_per_set
        self.n_trials = n_trials
        self.minibatch_size = minibatch_size
        self.full_eval_steps = full_eval_steps
        self.threshold = threshold
        self.rng_seed = rng_seed

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _build_demo_sets(
        self,
        pool: dict[str, list[Demo]],
        module_names: list[str],
        rng: random.Random,
    ) -> dict[str, list[list[Demo]]]:
        """Set 0 is empty; sets 1..n-1 are random subsets of the pool (deduped)."""
        demo_sets: dict[str, list[list[Demo]]] = {}
        for name in module_names:
            sets: list[list[Demo]] = [[]]
            seen: set[tuple] = {()}
            mod_pool = pool.get(name, [])
            for _ in range(1, self.n_demo_sets):
                if not mod_pool:
                    continue
                k = min(self.demos_per_set, len(mod_pool))
                chosen = rng.sample(mod_pool, k)
                key = tuple(
                    (tuple(sorted(d.inputs.items())), tuple(sorted(d.outputs.items())))
                    for d in chosen
                )
                if key in seen:
                    continue
                seen.add(key)
                sets.append(chosen)
            demo_sets[name] = sets
        return demo_sets

    async def _summarize_dataset(
        self,
        trainset: list[Example],
        budget: Budget,
        harness: EvalHarness,
        proposer_model: str,
        rng: random.Random,
    ) -> str:
        """One LLM call summarising up to 10 sampled examples (cost-charged)."""
        if budget.exhausted:
            return ""
        n = min(10, len(trainset))
        samples = rng.sample(trainset, n) if n < len(trainset) else list(trainset)
        prompt = _build_dataset_summary_prompt(samples)
        call = LLMCall(
            model=proposer_model,
            messages=(Message(role="user", content=prompt),),
            temperature=0.0,
            max_tokens=harness.cfg.max_tokens,
            seed=90_000,
        )
        resp = await harness.client.complete(call)
        budget.add_cost(resp.cost_usd)
        return resp.text.strip()

    async def _propose_instructions(
        self,
        program: PromptProgram,
        seed: Candidate,
        pool: dict[str, list[Demo]],
        dataset_summary: str,
        program_summary: str,
        budget: Budget,
        harness: EvalHarness,
        proposer_model: str,
        rng: random.Random,
        emit: Callable[[RunEvent], None],
    ) -> dict[str, list[str]]:
        """Per module: option 0 is the seed instruction, plus grounded proposals."""
        instructions: dict[str, list[str]] = {}
        for mod_idx, module in enumerate(program.modules):
            current = seed.modules[module.name].instruction
            options = [current]
            batch_proposals: list[str] = []
            for i in range(1, self.n_instruction_candidates):
                if budget.exhausted:
                    break
                tip = rng.choice(TIPS)
                prompt = _build_proposal_prompt(
                    dataset_summary,
                    program_summary,
                    module.name,
                    current,
                    pool.get(module.name, []),
                    batch_proposals,
                    tip,
                )
                call = LLMCall(
                    model=proposer_model,
                    messages=(Message(role="user", content=prompt),),
                    temperature=1.0,
                    max_tokens=harness.cfg.max_tokens,
                    seed=91_000 + mod_idx * 100 + i,
                )
                resp = await harness.client.complete(call)
                budget.add_cost(resp.cost_usd)
                proposed = parse_new_instruction(resp.text)
                options.append(proposed)
                batch_proposals.append(proposed)
                # Derive a stable id from content so the TUI lineage tree can
                # render MIPRO proposals.  SHA-256 of "module:instruction"
                # gives a collision-free 12-hex-char id without any new deps.
                proposal_id = hashlib.sha256(
                    f"{module.name}:{proposed}".encode()
                ).hexdigest()[:12]
                emit(
                    RunEvent.now(
                        "candidate_proposed",
                        candidate_id=proposal_id,
                        module=module.name,
                        tip=tip,
                        instruction=proposed,
                    )
                )
            instructions[module.name] = options
        return instructions

    # ------------------------------------------------------------------
    # optimize
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
        events: list[RunEvent] = []

        def _emit(event: RunEvent) -> None:
            events.append(event)
            emit(event)

        _emit(RunEvent.now("run_started", optimizer=self.name))

        rng = random.Random(self.rng_seed)
        module_names = [m.name for m in program.modules]
        proposer_model = harness.cfg.reflection_model or harness.cfg.task_model

        # ----------------------------------------------------------------
        # Stage 1: bootstrap a demo pool and build demo-set options.
        # ----------------------------------------------------------------
        max_pool = self.demos_per_set * max(0, self.n_demo_sets - 1)
        shuffled = list(trainset)
        rng.shuffle(shuffled)
        if max_pool > 0:
            pool, _, _ = await collect_demo_pool(
                program,
                seed,
                shuffled,
                metric,
                budget,
                harness,
                threshold=self.threshold,
                max_per_module=max_pool,
            )
        else:
            pool = {name: [] for name in module_names}
        demo_sets = self._build_demo_sets(pool, module_names, rng)

        # ----------------------------------------------------------------
        # Stage 2: grounded instruction proposal.
        # ----------------------------------------------------------------
        dataset_summary = ""
        if self.n_instruction_candidates > 1:
            dataset_summary = await self._summarize_dataset(
                trainset, budget, harness, proposer_model, rng
            )
        program_summary = _build_program_summary(program, seed)
        instructions = await self._propose_instructions(
            program,
            seed,
            pool,
            dataset_summary,
            program_summary,
            budget,
            harness,
            proposer_model,
            rng,
            _emit,
        )

        # ----------------------------------------------------------------
        # Stage 3: TPE search over (instruction, demo set) per module.
        # ----------------------------------------------------------------
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.rng_seed),
        )

        mb_scores: dict[_ConfigKey, float] = {}
        cand_by_key: dict[_ConfigKey, Candidate] = {}
        full_scores: dict[_ConfigKey, float] = {}
        all_candidates: list[Candidate] = [seed]
        completed = 0

        async def _full_eval_best_pending() -> None:
            """Full-eval the best-minibatch config not yet full-evaluated."""
            pending = [k for k in mb_scores if k not in full_scores]
            if not pending or budget.exhausted:
                return
            key = max(pending, key=lambda k: mb_scores[k])
            candidate = cand_by_key[key]
            report = await harness.evaluate(program, candidate, trainset, metric, budget)
            if report.truncated:
                # Budget ran out mid-eval: the mean is unreliable — skip it.
                return
            full_scores[key] = report.mean_score
            _emit(
                RunEvent.now(
                    "full_eval",
                    candidate_id=candidate.id,
                    config=candidate.meta,
                    mean_score=report.mean_score,
                )
            )

        for trial_idx in range(self.n_trials):
            if budget.exhausted:
                break

            trial = study.ask()
            config_inst: dict[str, int] = {}
            config_demo: dict[str, int] = {}
            for name in module_names:
                config_inst[name] = trial.suggest_categorical(
                    f"inst_{name}", list(range(len(instructions[name])))
                )
                config_demo[name] = trial.suggest_categorical(
                    f"demo_{name}", list(range(len(demo_sets[name])))
                )
            key: _ConfigKey = (
                tuple(sorted(config_inst.items())),
                tuple(sorted(config_demo.items())),
            )

            cached = key in mb_scores
            if cached:
                score = mb_scores[key]
            else:
                modules = {
                    name: ModuleState(
                        instruction=instructions[name][config_inst[name]],
                        demos=list(demo_sets[name][config_demo[name]]),
                    )
                    for name in module_names
                }
                candidate = seed.child(modules=modules, optimizer=self.name)
                candidate.meta = {"inst": dict(config_inst), "demo": dict(config_demo)}

                if self.minibatch_size < len(trainset):
                    batch = rng.sample(trainset, self.minibatch_size)
                else:
                    batch = list(trainset)
                report = await harness.evaluate(program, candidate, batch, metric, budget)
                if report.truncated:
                    # Budget ran out mid-minibatch: the mean is unreliable.
                    # Tell the study the trial was pruned and drop the config.
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    _emit(
                        RunEvent.now(
                            "budget_tick",
                            trial=trial_idx,
                            rollouts_used=budget.rollouts_used,
                            cost_used=budget.cost_used,
                        )
                    )
                    continue
                score = report.mean_score
                mb_scores[key] = score
                all_candidates.append(candidate)
                cand_by_key[key] = candidate

            study.tell(trial, score)
            completed += 1
            _emit(
                RunEvent.now(
                    "minibatch_scored",
                    trial=trial_idx,
                    config={"inst": dict(config_inst), "demo": dict(config_demo)},
                    score=score,
                    cached=cached,
                )
            )
            _emit(
                RunEvent.now(
                    "budget_tick",
                    trial=trial_idx,
                    rollouts_used=budget.rollouts_used,
                    cost_used=budget.cost_used,
                )
            )

            if completed % self.full_eval_steps == 0:
                await _full_eval_best_pending()

        # ----------------------------------------------------------------
        # Final selection: prefer full-eval scores.
        # ----------------------------------------------------------------
        if not full_scores and mb_scores and not budget.exhausted:
            await _full_eval_best_pending()

        if full_scores:
            best_key = max(full_scores, key=lambda k: full_scores[k])
            best = cand_by_key[best_key]
        elif mb_scores:
            best_key = max(mb_scores, key=lambda k: mb_scores[k])
            best = cand_by_key[best_key]
        else:
            best = seed

        scores: dict[str, float] = {
            cand_by_key[k].id: full_scores.get(k, mb_scores[k]) for k in cand_by_key
        }
        if best.id not in scores:
            scores[best.id] = 0.0

        _emit(RunEvent.now("run_finished", optimizer=self.name, best_id=best.id))

        return OptimizeResult(
            best=best,
            candidates=all_candidates,
            scores=scores,
            events_count=len(events),
        )
