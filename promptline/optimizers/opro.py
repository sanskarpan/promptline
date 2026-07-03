"""OPRO — Optimization by PROmpting.

Iteratively proposes new instructions by showing the LLM a trajectory of
(instruction, score) pairs sorted *ascending* by score (best last) and asking
it to write a better one.

.. note::
    This optimizer requires a strong proposer model capable of meta-level
    reasoning about task instructions.  See arXiv:2309.03409 and the follow-up
    analysis at arXiv:2405.10276 for guidance on model selection.
"""
from __future__ import annotations

import random
import re
from collections.abc import Callable

from promptline.core.llm import LLMCall, Message
from promptline.core.program import PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, Metric
from promptline.optimizers.base import Optimizer, OptimizeResult, RunEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INS_RE = re.compile(r"<INS>(.*?)</INS>", re.DOTALL)


def _parse_instruction(text: str) -> str:
    """Extract the first <INS>…</INS> block; fall back to the stripped reply."""
    m = _INS_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _build_meta_prompt(seed_context: str, trajectory: list[tuple[str, float]]) -> str:
    """Construct the meta-prompt sent to the proposer model.

    Parameters
    ----------
    seed_context:
        The original task instruction used as background context.
    trajectory:
        List of (instruction, score) pairs.  Must already be sorted
        *ascending* by score (worst first, best last).
    """
    lines: list[str] = [
        "You are an expert prompt engineer.",
        "",
        "Task context (seed instruction):",
        seed_context,
        "",
        "Below is the optimization trajectory so far, sorted from worst to best:",
    ]
    for instruction, score in trajectory:
        lines.append(f'score={score:.4f}: "{instruction}"')
    lines += [
        "",
        "Write a new instruction that achieves a higher score than all of the above.",
        "Wrap it in <INS></INS> tags.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class OPRO(Optimizer):
    """Optimization by PROmpting (OPRO).

    Parameters
    ----------
    n_steps:
        Maximum number of optimization steps (early-stopped by budget).
    candidates_per_step:
        Number of new instruction proposals generated at each step.
    minibatch_size:
        If set, evaluate proposals on a random minibatch of this size instead
        of the full training set.
    max_trajectory:
        Maximum number of (instruction, score) entries kept in the trajectory.
        Entries with the highest scores are retained when the cap is reached.
    rng_seed:
        Seed for minibatch sampling.

    Notes
    -----
    Only single-module programs are fully supported.  For multi-module
    programs the new instruction is applied to the *first* module only.
    """

    name = "opro"

    def __init__(
        self,
        n_steps: int = 10,
        candidates_per_step: int = 4,
        minibatch_size: int | None = None,
        max_trajectory: int = 20,
        rng_seed: int = 0,
    ) -> None:
        self.n_steps = n_steps
        self.candidates_per_step = candidates_per_step
        self.minibatch_size = minibatch_size
        self.max_trajectory = max_trajectory
        self.rng_seed = rng_seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_candidate(self, seed: Candidate, new_instruction: str) -> Candidate:
        """Create a child candidate with *new_instruction* on the first module."""
        first_mod = next(iter(seed.modules))
        new_modules: dict[str, ModuleState] = {}
        for mod_name, state in seed.modules.items():
            if mod_name == first_mod:
                new_modules[mod_name] = ModuleState(
                    instruction=new_instruction,
                    demos=list(state.demos),
                )
            else:
                new_modules[mod_name] = ModuleState(
                    instruction=state.instruction,
                    demos=list(state.demos),
                )
        return seed.child(modules=new_modules, optimizer=self.name)

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

        # Determine seed context from the first module's original instruction.
        first_mod_name = program.modules[0].name if program.modules else ""
        seed_context = seed.modules[first_mod_name].instruction if first_mod_name else ""

        # Proposer model: prefer reflection_model, fall back to task_model.
        proposer_model = harness.cfg.reflection_model or harness.cfg.task_model

        # ----------------------------------------------------------------
        # Step 0: evaluate seed so it appears in trajectory and scores.
        # ----------------------------------------------------------------
        def _pick_examples() -> list:
            if self.minibatch_size is not None and self.minibatch_size < len(trainset):
                return rng.sample(trainset, self.minibatch_size)
            return trainset

        seed_batch = _pick_examples()
        seed_report = await harness.evaluate(
            program, seed, seed_batch, metric, budget
        )
        seed_score = seed_report.mean_score

        all_candidates: list[Candidate] = [seed]
        all_scores: dict[str, float] = {seed.id: seed_score}

        # trajectory: list of (instruction, score), kept ≤ max_trajectory
        trajectory: list[tuple[str, float]] = [(seed_context, seed_score)]

        _emit(
            RunEvent.now(
                "minibatch_scored",
                candidate_id=seed.id,
                mean_score=seed_score,
                step=0,
            )
        )

        # ----------------------------------------------------------------
        # Main loop
        # ----------------------------------------------------------------
        for step in range(1, self.n_steps + 1):
            if budget.exhausted:
                break

            # Sort trajectory ascending by score (best last for the LLM).
            traj_sorted = sorted(trajectory, key=lambda t: t[1])

            meta_prompt = _build_meta_prompt(seed_context, traj_sorted)
            meta_messages = (
                Message(role="user", content=meta_prompt),
            )

            # Propose candidates_per_step new instructions.
            for _ in range(self.candidates_per_step):
                if budget.exhausted:
                    break

                llm_call = LLMCall(
                    model=proposer_model,
                    messages=meta_messages,
                    temperature=1.0,
                    max_tokens=harness.cfg.max_tokens,
                )
                resp = await harness.client.complete(llm_call)
                budget.add_cost(resp.cost_usd)

                new_instruction = _parse_instruction(resp.text)
                candidate = self._make_candidate(seed, new_instruction)
                all_candidates.append(candidate)

                _emit(
                    RunEvent.now(
                        "candidate_proposed",
                        candidate_id=candidate.id,
                        step=step,
                        instruction=new_instruction,
                    )
                )

                # Evaluate candidate on minibatch or full trainset.
                batch = _pick_examples()
                report = await harness.evaluate(
                    program, candidate, batch, metric, budget
                )
                cand_score = report.mean_score
                all_scores[candidate.id] = cand_score

                _emit(
                    RunEvent.now(
                        "minibatch_scored",
                        candidate_id=candidate.id,
                        mean_score=cand_score,
                        step=step,
                    )
                )

                # Update trajectory.
                trajectory.append((new_instruction, cand_score))
                if len(trajectory) > self.max_trajectory:
                    # Keep the top-scoring entries.
                    trajectory = sorted(
                        trajectory, key=lambda t: t[1], reverse=True
                    )[: self.max_trajectory]

        # ----------------------------------------------------------------
        # Pick best
        # ----------------------------------------------------------------
        best_id = max(all_scores, key=lambda cid: all_scores[cid])
        best_candidate = next(c for c in all_candidates if c.id == best_id)

        _emit(
            RunEvent.now(
                "run_finished",
                optimizer=self.name,
                best_id=best_id,
            )
        )

        return OptimizeResult(
            best=best_candidate,
            candidates=all_candidates,
            scores=all_scores,
            events_count=len(events),
        )
