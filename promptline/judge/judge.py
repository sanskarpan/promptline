"""LLM-as-judge: rubric-based pointwise and pairwise judges.

Both judges are built on :class:`~promptline.core.program.PromptProgram`, so
their instructions live in a :class:`~promptline.core.types.Candidate` and can
be meta-optimized like any other program (see
:mod:`promptline.judge.calibrator`).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from promptline.core.llm import LLMCall, LLMClient, LLMResponse
from promptline.core.program import ModelConfig, Prediction, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.data.dataset import Record
from promptline.eval.harness import Metric, MetricResult

# ---------------------------------------------------------------------------
# Errors and value objects
# ---------------------------------------------------------------------------


class JudgeError(Exception):
    """Raised when a judge cannot produce a usable score or verdict."""


class RubricCriterion(BaseModel):
    """A single evaluation criterion with an integer scale and anchors."""

    name: str
    description: str
    scale: tuple[int, int] = (1, 5)
    anchors: dict[int, str] = {}


class JudgeScore(BaseModel):
    """Result of a pointwise judgement (possibly averaged over k samples)."""

    value: float
    reasoning: str
    raw: list[float]


class PairwiseVerdict(BaseModel):
    """Result of a position-debiased pairwise comparison."""

    winner: Literal["A", "B", "TIE"]
    reasoning: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JUDGE_MODULE = "judge"


def render_transcript(record: Record) -> str:
    """Render a Record's conversation as a role-prefixed transcript string."""
    return "\n".join(f"{t.role}: {t.content}" for t in record.conversation)


def _scale_lines(criterion: RubricCriterion) -> list[str]:
    lo, hi = criterion.scale
    lines = [f"Rate on an integer scale from {lo} (worst) to {hi} (best)."]
    if criterion.anchors:
        lines.append("Scale anchors:")
        for point in sorted(criterion.anchors):
            lines.append(f"- {point}: {criterion.anchors[point]}")
    return lines


def build_pointwise_instruction(criterion: RubricCriterion) -> str:
    """Default pointwise judge instruction for *criterion*."""
    lo, hi = criterion.scale
    lines = [
        "You are an impartial expert evaluator of assistant responses.",
        (f"Evaluate the response on the criterion '{criterion.name}': {criterion.description}"),
        *_scale_lines(criterion),
        "Reason step by step about the response quality first.",
        "Do not reward length or verbosity.",
        (
            "Answer with [[reasoning]]: your step-by-step analysis, then "
            f"[[score]]: a single integer between {lo} and {hi}."
        ),
    ]
    return "\n".join(lines)


def build_pairwise_instruction(criterion: RubricCriterion) -> str:
    """Default pairwise judge instruction for *criterion*."""
    lines = [
        "You are an impartial expert evaluator comparing two assistant "
        "responses (A and B) to the same conversation.",
        (f"Compare them on the criterion '{criterion.name}': {criterion.description}"),
        "Reason step by step about the quality of both responses first.",
        "Do not reward length or verbosity.",
        "Do not let the order of presentation influence your judgement.",
        (
            "Answer with [[reasoning]]: your step-by-step analysis, then "
            "[[verdict]]: exactly one of A, B, or TIE."
        ),
    ]
    return "\n".join(lines)


class _SeededClient:
    """Wraps a client so every call carries a fixed LLMCall seed."""

    def __init__(self, inner: LLMClient, seed: int) -> None:
        self.inner = inner
        self.seed = seed

    async def complete(self, call: LLMCall) -> LLMResponse:
        return await self.inner.complete(call.model_copy(update={"seed": self.seed}))


# ---------------------------------------------------------------------------
# Pointwise judge
# ---------------------------------------------------------------------------


class PointwiseJudge:
    """Scores a single response against a rubric criterion.

    The judge prompt is a one-module :class:`PromptProgram`; ``seed_candidate``
    holds the default instruction so a calibrator can meta-optimize it and
    pass the optimized candidate back into :meth:`score`.
    """

    def __init__(
        self,
        criterion: RubricCriterion,
        judge_model: str,
        samples: int = 1,
        temperature_when_sampling: float = 0.7,
    ) -> None:
        self.criterion = criterion
        self.judge_model = judge_model
        self.samples = samples
        self.temperature_when_sampling = temperature_when_sampling

        instruction = build_pointwise_instruction(criterion)
        self.program = PromptProgram.simple(
            instruction=instruction,
            inputs=["conversation", "response", "reference"],
            outputs=["reasoning", "score"],
            name=JUDGE_MODULE,
        )
        self.seed_candidate = Candidate.seed({JUDGE_MODULE: ModuleState(instruction=instruction)})

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_score(self, text: str) -> int | None:
        """Extract the first integer from *text*, clamped to the scale."""
        match = re.search(r"-?\d+", text)
        if match is None:
            return None
        lo, hi = self.criterion.scale
        return max(lo, min(hi, int(match.group())))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def _score_inputs(
        self,
        inputs: dict[str, str],
        client: LLMClient,
        candidate: Candidate | None,
    ) -> JudgeScore:
        cand = candidate or self.seed_candidate
        example = Example(inputs=inputs)

        # samples=1 -> one deterministic call; samples>1 -> k sampled calls
        # with distinct seeds so a caching client still sees distinct keys.
        runs: list[tuple[ModelConfig, LLMClient]]
        if self.samples <= 1:
            cfg = ModelConfig(task_model=self.judge_model, temperature=0.0)
            runs = [(cfg, client)]
        else:
            cfg = ModelConfig(
                task_model=self.judge_model,
                temperature=self.temperature_when_sampling,
            )
            runs = [(cfg, _SeededClient(client, s)) for s in range(self.samples)]

        raw: list[float] = []
        reasoning = ""
        for run_cfg, run_client in runs:
            prediction = await self.program.run(example, cand, run_client, run_cfg)
            if prediction.failed:
                continue  # drop this sample
            value = self.parse_score(prediction.outputs.get("score", ""))
            if value is None:
                continue  # drop this sample
            raw.append(float(value))
            if not reasoning:
                reasoning = prediction.outputs.get("reasoning", "")

        if not raw:
            raise JudgeError(
                f"judge produced no parseable score in {len(runs)} sample(s) "
                f"for criterion '{self.criterion.name}'"
            )
        return JudgeScore(value=sum(raw) / len(raw), reasoning=reasoning, raw=raw)

    async def score(
        self,
        record: Record,
        response: str,
        client: LLMClient,
        candidate: Candidate | None = None,
        reference: str | None = None,
    ) -> JudgeScore:
        """Judge *response* to *record*'s conversation on the criterion."""
        inputs = {"conversation": render_transcript(record), "response": response}
        if reference is not None:
            inputs["reference"] = reference
        return await self._score_inputs(inputs, client, candidate)

    # ------------------------------------------------------------------
    # Metric adapter
    # ------------------------------------------------------------------

    def as_metric(
        self,
        client: LLMClient,
        reference_from_labels: bool = True,
        candidate: Candidate | None = None,
    ) -> Metric:
        """Adapt this judge into a harness :data:`Metric`.

        Scores the program's ``answer``/``response`` (or last) output field and
        normalizes the judge value onto [0, 1] via ``(v - lo) / (hi - lo)``.

        *candidate* is forwarded to :meth:`_score_inputs`; pass an optimized
        candidate to use its instruction instead of the seed.

        The metric **never raises**: any :class:`JudgeError` or unexpected
        exception is caught and returned as ``MetricResult(score=0.0)``.
        """
        lo, hi = self.criterion.scale

        async def metric(example: Example, prediction: Prediction) -> MetricResult:
            outputs = prediction.outputs
            response = (
                outputs.get("answer")
                or outputs.get("response")
                or (next(reversed(outputs.values()), "") if outputs else "")
            )
            inputs = {
                "conversation": example.inputs.get("conversation", ""),
                "response": response,
            }
            if reference_from_labels:
                reference = example.labels.get("reference")
                if reference is not None:
                    inputs["reference"] = reference
            try:
                judged = await self._score_inputs(inputs, client, candidate)
                return MetricResult(
                    score=(judged.value - lo) / (hi - lo),
                    feedback=judged.reasoning,
                )
            except (JudgeError, Exception) as exc:
                return MetricResult(score=0.0, feedback=f"judge error: {exc}")

        return metric


# ---------------------------------------------------------------------------
# Pairwise judge
# ---------------------------------------------------------------------------

# TIE is matched case-insensitively; single-letter A/B verdicts are matched
# case-SENSITIVELY so lowercase prose articles ("a"/"b") are not misread as a
# verdict (judges are instructed to emit uppercase A/B/TIE).
_TIE_RE = re.compile(r"\bTIE\b", re.IGNORECASE)
_AB_RE = re.compile(r"\b(A|B)\b")

Verdict = Literal["A", "B", "TIE"]

_VERDICTS: dict[str, Verdict] = {"A": "A", "B": "B", "TIE": "TIE"}

_UNSWAP: dict[Verdict, Verdict] = {"A": "B", "B": "A", "TIE": "TIE"}


class PairwiseJudge:
    """Compares two responses with position-debiasing (both orderings)."""

    def __init__(self, criterion: RubricCriterion, judge_model: str) -> None:
        self.criterion = criterion
        self.judge_model = judge_model

        instruction = build_pairwise_instruction(criterion)
        self.program = PromptProgram.simple(
            instruction=instruction,
            inputs=["conversation", "response_a", "response_b"],
            outputs=["reasoning", "verdict"],
            name=JUDGE_MODULE,
        )
        self.seed_candidate = Candidate.seed({JUDGE_MODULE: ModuleState(instruction=instruction)})

    @staticmethod
    def parse_verdict(text: str) -> Verdict | None:
        """Parse a verdict from *text*.

        Prefers an explicit TIE (case-insensitive); otherwise the first
        standalone case-sensitive ``A``/``B`` token; otherwise ``None``.  This
        avoids misreading lowercase prose articles ("a"/"b") as verdicts.
        """
        if _TIE_RE.search(text):
            return "TIE"
        match = _AB_RE.search(text)
        return _VERDICTS[match.group(1)] if match else None

    async def _one_ordering(
        self,
        transcript: str,
        first: str,
        second: str,
        client: LLMClient,
        candidate: Candidate,
    ) -> tuple[Verdict | None, str]:
        cfg = ModelConfig(task_model=self.judge_model, temperature=0.0)
        example = Example(
            inputs={
                "conversation": transcript,
                "response_a": first,
                "response_b": second,
            }
        )
        prediction = await self.program.run(example, candidate, client, cfg)
        if prediction.failed:
            return None, ""
        verdict = self.parse_verdict(prediction.outputs.get("verdict", ""))
        return verdict, prediction.outputs.get("reasoning", "")

    async def compare(
        self,
        record: Record,
        response_a: str,
        response_b: str,
        client: LLMClient,
        candidate: Candidate | None = None,
    ) -> PairwiseVerdict:
        """Judge both orderings; agreement wins, disagreement is a TIE."""
        cand = candidate or self.seed_candidate
        transcript = render_transcript(record)

        verdict_1, reasoning_1 = await self._one_ordering(
            transcript, response_a, response_b, client, cand
        )
        verdict_2_swapped, reasoning_2 = await self._one_ordering(
            transcript, response_b, response_a, client, cand
        )
        if verdict_1 is None or verdict_2_swapped is None:
            raise JudgeError(
                f"judge produced no parseable verdict for criterion '{self.criterion.name}'"
            )
        verdict_2 = _UNSWAP[verdict_2_swapped]
        if verdict_1 == verdict_2:
            winner = verdict_1
            reasoning = reasoning_1
        else:
            winner = "TIE"
            reasoning = (
                f"Position-swap disagreement: [A-order] {reasoning_1} | [B-order] {reasoning_2}"
            )
        return PairwiseVerdict(winner=winner, reasoning=reasoning)
