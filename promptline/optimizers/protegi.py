"""ProTeGi — Prompt optimization with Textual Gradients (EMNLP 2023).

Beam search over instructions.  Each round, every beam candidate is run on a
minibatch; its *failing* examples are fed to an LLM which writes a natural-
language critique (the "textual gradient"), a second LLM call edits the
instruction to fix the diagnosed problem, and paraphrase calls expand each
edit.  The pooled parents + children are then pruned back to the beam with
CAPO-style successive-halving racing (arXiv:2504.16005): survivors are scored
on fresh racing batches and the bottom half is dropped each racing round, so
selection costs far fewer rollouts than full-evaluating every candidate.

See arXiv:2305.03495 (ProTeGi) and arXiv:2504.16005 (CAPO).
"""
from __future__ import annotations

import math
import random
import re
from collections.abc import Callable

from promptline.core.llm import LLMCall, Message
from promptline.core.program import PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, EvalReport, Metric
from promptline.optimizers.base import Optimizer, OptimizeResult, RunEvent

# ---------------------------------------------------------------------------
# Prompt templates and parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:[\w-]*)\n(.*?)```", re.DOTALL)

#: Maximum number of failing examples included in a gradient prompt.
_MAX_FAILURES = 4

#: Maximum characters of example inputs quoted per failure.
_INPUT_EXCERPT = 300


def _parse_fenced(text: str) -> str:
    """Extract the first fenced code block; fall back to the whole reply."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _format_failures(batch: list[Example], report: EvalReport) -> str:
    """Render up to :data:`_MAX_FAILURES` failing examples for the gradient prompt."""
    lines: list[str] = []
    shown = 0
    for res in report.per_example:
        if res.score >= 1.0:
            continue
        example = batch[res.example_idx]
        inputs = "; ".join(f"{k}: {v}" for k, v in example.inputs.items())
        lines.append(f"Example inputs: {inputs[:_INPUT_EXCERPT]}")
        lines.append(f"Score: {res.score:.4f}")
        lines.append(f"Feedback: {res.feedback}")
        lines.append("")
        shown += 1
        if shown >= _MAX_FAILURES:
            break
    return "\n".join(lines).rstrip()


def _gradient_prompt(instruction: str, failures: str) -> str:
    return (
        "The following instruction is used to prompt a language model:\n\n"
        f"Instruction:\n{instruction}\n\n"
        "It produced these failing examples:\n\n"
        f"{failures}\n\n"
        "In 2-3 sentences, diagnose why this instruction failed on these examples."
    )


def _edit_prompt(instruction: str, gradient: str) -> str:
    return (
        f"Instruction:\n{instruction}\n\n"
        f"Diagnosis of its failures:\n{gradient}\n\n"
        "Rewrite the instruction to fix this problem. "
        "Output ONLY the new instruction in a fenced code block."
    )


def _paraphrase_prompt(instruction: str) -> str:
    return (
        f"Instruction:\n{instruction}\n\n"
        "Paraphrase this instruction keeping its meaning. "
        "Output ONLY the paraphrase in a fenced code block."
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class ProTeGi(Optimizer):
    """Textual-gradient prompt optimizer with successive-halving racing.

    Parameters
    ----------
    beam_width:
        Number of candidates kept in the beam between rounds.
    n_gradients:
        Textual-gradient (critique) LLM calls per failing beam candidate;
        each gradient yields one edited child.
    n_paraphrases:
        Paraphrase expansions generated for each edited child.
    n_rounds:
        Optimization rounds (early-stopped on budget exhaustion).
    minibatch_size:
        Examples per gradient minibatch used to collect failures.
    racing_rounds:
        Maximum successive-halving rounds when pruning the pool.
    racing_batch:
        Examples per racing batch (fresh, disjoint per racing round).
    rng_seed:
        Seed for minibatch and racing-batch sampling.

    Notes
    -----
    Only single-module programs are fully supported.  For multi-module
    programs the new instruction is applied to the *first* module only.
    """

    name = "protegi"

    def __init__(
        self,
        beam_width: int = 4,
        n_gradients: int = 2,
        n_paraphrases: int = 1,
        n_rounds: int = 3,
        minibatch_size: int = 8,
        racing_rounds: int = 3,
        racing_batch: int = 8,
        rng_seed: int = 0,
    ) -> None:
        self.beam_width = beam_width
        self.n_gradients = n_gradients
        self.n_paraphrases = n_paraphrases
        self.n_rounds = n_rounds
        self.minibatch_size = minibatch_size
        self.racing_rounds = racing_rounds
        self.racing_batch = racing_batch
        self.rng_seed = rng_seed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_candidate(self, parent: Candidate, new_instruction: str) -> Candidate:
        """Child of *parent* with the first module's instruction replaced."""
        first_mod = next(iter(parent.modules))
        modules: dict[str, ModuleState] = {}
        for name, state in parent.modules.items():
            if name == first_mod:
                modules[name] = ModuleState(
                    instruction=new_instruction, demos=list(state.demos)
                )
            else:
                modules[name] = state.model_copy(deep=True)
        return parent.child(modules=modules, optimizer=self.name)

    @staticmethod
    def _first_instruction(candidate: Candidate) -> str:
        return next(iter(candidate.modules.values())).instruction

    async def _propose(
        self,
        prompt: str,
        harness: EvalHarness,
        budget: Budget,
        llm_seed: int,
    ) -> str:
        """One proposer LLM call: charge cost only, distinct seed per call."""
        model = harness.cfg.reflection_model or harness.cfg.task_model
        call = LLMCall(
            model=model,
            messages=(Message(role="user", content=prompt),),
            temperature=1.0,
            max_tokens=harness.cfg.max_tokens,
            # Distinct seed per proposer call so a caching client keyed on
            # LLMCall.key() cannot collapse repeated prompts into one response.
            seed=llm_seed,
        )
        resp = await harness.client.complete(call)
        budget.add_cost(resp.cost_usd)
        return resp.text

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
        events_count = 0

        def _emit(event: RunEvent) -> None:
            nonlocal events_count
            events_count += 1
            emit(event)

        _emit(RunEvent.now("run_started", optimizer=self.name, n_rounds=self.n_rounds))

        rng = random.Random(self.rng_seed)
        llm_seed = 0  # monotonically increasing seed for proposer calls

        beam: list[Candidate] = [seed]
        all_candidates: list[Candidate] = [seed]
        minibatch_scores: dict[str, float] = {}
        racing_scores: dict[str, list[float]] = {}

        def _acc_mean(candidate: Candidate) -> float:
            raced = racing_scores.get(candidate.id)
            if raced:
                return sum(raced) / len(raced)
            return minibatch_scores.get(candidate.id, 0.0)

        for round_no in range(1, self.n_rounds + 1):
            if budget.exhausted:
                break
            _emit(
                RunEvent.now(
                    "budget_tick",
                    round=round_no,
                    rollouts_used=budget.rollouts_used,
                    cost_used=budget.cost_used,
                    max_rollouts=budget.max_rollouts,
                    max_cost_usd=budget.max_cost_usd,
                )
            )

            # ------------------------------------------------------------
            # 1. Expansion: gradients -> edits -> paraphrases per beam parent.
            # ------------------------------------------------------------
            children: list[Candidate] = []
            for parent in beam:
                if budget.exhausted:
                    break
                batch = rng.sample(
                    trainset, min(self.minibatch_size, len(trainset))
                )
                report = await harness.evaluate(program, parent, batch, metric, budget)
                minibatch_scores[parent.id] = report.mean_score
                _emit(
                    RunEvent.now(
                        "minibatch_scored",
                        candidate_id=parent.id,
                        mean_score=report.mean_score,
                        round=round_no,
                        phase="minibatch",
                    )
                )

                failures = _format_failures(batch, report)
                if not failures:
                    continue  # no failing examples: this parent emits no gradients

                instruction = self._first_instruction(parent)
                for _ in range(self.n_gradients):
                    if budget.exhausted:
                        break
                    llm_seed += 1
                    gradient = await self._propose(
                        _gradient_prompt(instruction, failures),
                        harness,
                        budget,
                        llm_seed,
                    )

                    llm_seed += 1
                    edit_reply = await self._propose(
                        _edit_prompt(instruction, gradient),
                        harness,
                        budget,
                        llm_seed,
                    )
                    new_instruction = _parse_fenced(edit_reply)
                    child = self._make_candidate(parent, new_instruction)
                    children.append(child)
                    all_candidates.append(child)
                    _emit(
                        RunEvent.now(
                            "candidate_proposed",
                            candidate_id=child.id,
                            parent_id=parent.id,
                            round=round_no,
                            source="gradient",
                            instruction=new_instruction,
                        )
                    )

                    for _ in range(self.n_paraphrases):
                        if budget.exhausted:
                            break
                        llm_seed += 1
                        para_reply = await self._propose(
                            _paraphrase_prompt(new_instruction),
                            harness,
                            budget,
                            llm_seed,
                        )
                        para_instruction = _parse_fenced(para_reply)
                        para_child = self._make_candidate(child, para_instruction)
                        children.append(para_child)
                        all_candidates.append(para_child)
                        _emit(
                            RunEvent.now(
                                "candidate_proposed",
                                candidate_id=para_child.id,
                                parent_id=child.id,
                                round=round_no,
                                source="paraphrase",
                                instruction=para_instruction,
                            )
                        )

            # ------------------------------------------------------------
            # 2. Racing (successive halving) to select the next beam.
            # ------------------------------------------------------------
            pool = beam + children
            survivors = list(pool)

            # Disjoint racing batches: shuffle once, consume successive
            # slices; reshuffle when the remainder is too small.
            shuffled = list(trainset)
            rng.shuffle(shuffled)
            cursor = 0

            def _next_batch() -> list[Example]:
                nonlocal shuffled, cursor
                size = min(self.racing_batch, len(trainset))
                if cursor + size > len(shuffled):
                    shuffled = list(trainset)
                    rng.shuffle(shuffled)
                    cursor = 0
                batch = shuffled[cursor : cursor + size]
                cursor += size
                return batch

            for racing_round in range(1, self.racing_rounds + 1):
                if len(survivors) <= self.beam_width or budget.exhausted:
                    break
                batch = _next_batch()
                for cand in survivors:
                    if budget.exhausted:
                        break
                    report = await harness.evaluate(
                        program, cand, batch, metric, budget
                    )
                    racing_scores.setdefault(cand.id, []).append(report.mean_score)
                    _emit(
                        RunEvent.now(
                            "minibatch_scored",
                            candidate_id=cand.id,
                            mean_score=report.mean_score,
                            round=round_no,
                            phase="racing",
                            racing_round=racing_round,
                        )
                    )
                keep = max(self.beam_width, math.ceil(len(survivors) / 2))
                survivors = sorted(survivors, key=_acc_mean, reverse=True)[:keep]

            beam = sorted(survivors, key=_acc_mean, reverse=True)[: self.beam_width]

        # ----------------------------------------------------------------
        # Final scores: mean racing score, falling back to minibatch score.
        # ----------------------------------------------------------------
        all_scores = {c.id: _acc_mean(c) for c in all_candidates}
        best = beam[0] if beam else seed

        _emit(
            RunEvent.now(
                "run_finished",
                optimizer=self.name,
                best_id=best.id,
                n_candidates=len(all_candidates),
            )
        )

        return OptimizeResult(
            best=best,
            candidates=all_candidates,
            scores=all_scores,
            events_count=events_count,
        )
