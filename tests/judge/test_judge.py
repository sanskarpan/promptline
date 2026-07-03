"""Tests for promptline.judge.judge (Task 15)."""
from __future__ import annotations

import pytest

from promptline.core.llm import FakeLLMClient
from promptline.core.program import Prediction
from promptline.core.types import Example
from promptline.data.dataset import Record, Turn
from promptline.judge.judge import (
    JudgeError,
    PairwiseJudge,
    PointwiseJudge,
    RubricCriterion,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CRITERION = RubricCriterion(
    name="helpfulness",
    description="How well the response addresses the user's need.",
    scale=(1, 5),
    anchors={1: "Useless.", 3: "Partially helpful.", 5: "Fully addresses the need."},
)

RECORD = Record(conversation=[Turn(role="user", content="What is 2+2?")])


def _resp(score: str, reasoning: str = "step by step") -> str:
    return f"[[reasoning]]: {reasoning}\n[[score]]: {score}"


# ---------------------------------------------------------------------------
# PointwiseJudge — instruction template
# ---------------------------------------------------------------------------


def test_instruction_contains_criterion_and_anchors() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    instr = judge.seed_candidate.modules["judge"].instruction
    assert "helpfulness" in instr
    assert CRITERION.description in instr
    assert "Fully addresses the need." in instr
    assert "Partially helpful." in instr
    assert "Do not reward length or verbosity." in instr
    assert "Reason step by step" in instr
    assert "[[reasoning]]" in instr
    assert "[[score]]" in instr


def test_program_and_seed_candidate_exposed() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    assert judge.program.module_names == ["judge"]
    assert "judge" in judge.seed_candidate.modules


# ---------------------------------------------------------------------------
# PointwiseJudge — scoring
# ---------------------------------------------------------------------------


async def test_pointwise_parse_score() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_resp("4")])
    result = await judge.score(RECORD, "The answer is 4.", client)
    assert result.value == 4.0
    assert result.reasoning == "step by step"
    assert result.raw == [4.0]


async def test_pointwise_clamps_out_of_scale_score() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_resp("9")])
    result = await judge.score(RECORD, "resp", client)
    assert result.value == 5.0


async def test_pointwise_single_sample_uses_temperature_zero() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_resp("3")])
    await judge.score(RECORD, "resp", client)
    assert len(client.calls) == 1
    assert client.calls[0].temperature == 0.0
    assert client.calls[0].model == "fake/judge"


async def test_pointwise_k_sampling_averages_and_drops_unparseable() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge", samples=3)
    client = FakeLLMClient(script=[_resp("4"), _resp("N/A"), _resp("2")])
    result = await judge.score(RECORD, "resp", client)
    assert result.raw == [4.0, 2.0]
    assert result.value == 3.0


async def test_pointwise_k_sampling_distinct_seeds_and_temperature() -> None:
    judge = PointwiseJudge(
        criterion=CRITERION, judge_model="fake/judge", samples=3,
        temperature_when_sampling=0.7,
    )
    client = FakeLLMClient(script=[_resp("4"), _resp("3"), _resp("2")])
    await judge.score(RECORD, "resp", client)
    assert [c.seed for c in client.calls] == [0, 1, 2]
    assert all(c.temperature == 0.7 for c in client.calls)


async def test_pointwise_all_unparseable_raises_judge_error() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge", samples=2)
    client = FakeLLMClient(script=[_resp("N/A"), _resp("nope")])
    with pytest.raises(JudgeError):
        await judge.score(RECORD, "resp", client)


async def test_pointwise_reference_passed_into_prompt() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_resp("5")])
    await judge.score(RECORD, "resp", client, reference="GOLD-REF")
    user_msg = client.calls[0].messages[-1].content
    assert "GOLD-REF" in user_msg


# ---------------------------------------------------------------------------
# PointwiseJudge.as_metric
# ---------------------------------------------------------------------------


def _prediction(outputs: dict[str, str]) -> Prediction:
    return Prediction(outputs=outputs, traces=[], cost_usd=0.0)


async def test_as_metric_normalizes_max_to_one() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_resp("5")])
    metric = judge.as_metric(client)
    example = Example(inputs={"conversation": "user: hi"}, labels={"reference": "ref"})
    result = await metric(example, _prediction({"answer": "hello"}))
    assert result.score == 1.0
    assert result.feedback == "step by step"


async def test_as_metric_normalizes_min_to_zero() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_resp("1")])
    metric = judge.as_metric(client)
    example = Example(inputs={"conversation": "user: hi"})
    result = await metric(example, _prediction({"response": "hello"}))
    assert result.score == 0.0


async def test_as_metric_falls_back_to_last_output_field() -> None:
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_resp("3")])
    metric = judge.as_metric(client)
    example = Example(inputs={"conversation": "user: hi"})
    result = await metric(example, _prediction({"summary": "LAST-FIELD"}))
    assert result.score == 0.5
    assert "LAST-FIELD" in client.calls[0].messages[-1].content


async def test_as_metric_never_crashes_on_unparseable_judge() -> None:
    """Metric must return score=0.0 instead of raising when judge fails."""
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge", samples=2)
    # All responses are unparseable → _score_inputs raises JudgeError.
    client = FakeLLMClient(script=[_resp("N/A"), _resp("nope")])
    metric = judge.as_metric(client)
    example = Example(inputs={"conversation": "user: hi"})
    result = await metric(example, _prediction({"answer": "hello"}))
    assert result.score == 0.0
    assert "judge error" in result.feedback


async def test_as_metric_accepts_candidate_and_uses_its_instruction() -> None:
    """Optimized candidate's instruction must appear in the judge's system prompt."""
    from promptline.core.types import ModuleState
    from promptline.judge.judge import JUDGE_MODULE

    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_resp("4")])

    optimized_instruction = "CUSTOM-OPTIMIZED-INSTRUCTION-XYZ"
    from promptline.core.types import Candidate
    candidate = Candidate.seed({JUDGE_MODULE: ModuleState(instruction=optimized_instruction)})

    metric = judge.as_metric(client, candidate=candidate)
    example = Example(inputs={"conversation": "user: hi"})
    await metric(example, _prediction({"answer": "hello"}))

    # The instruction lands in the system prompt of the LLM call.
    system_msg = client.calls[0].messages[0].content
    assert optimized_instruction in system_msg


# ---------------------------------------------------------------------------
# PairwiseJudge
# ---------------------------------------------------------------------------

PAIR_CRITERION = RubricCriterion(name="quality", description="Overall quality.")


def _verdict(v: str, reasoning: str = "compared both") -> str:
    return f"[[reasoning]]: {reasoning}\n[[verdict]]: {v}"


async def test_pairwise_agreement_yields_winner() -> None:
    judge = PairwiseJudge(criterion=PAIR_CRITERION, judge_model="fake/judge")
    # Call 1 (a, b): A wins.  Call 2 (b, a): B wins => un-swapped A wins.
    client = FakeLLMClient(script=[_verdict("A"), _verdict("B")])
    verdict = await judge.compare(RECORD, "resp-a", "resp-b", client)
    assert verdict.winner == "A"
    assert verdict.reasoning == "compared both"


async def test_pairwise_swapped_ordering_actually_swaps_responses() -> None:
    judge = PairwiseJudge(criterion=PAIR_CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_verdict("A"), _verdict("B")])
    await judge.compare(RECORD, "RESP-A", "RESP-B", client)
    first = client.calls[0].messages[-1].content
    second = client.calls[1].messages[-1].content
    assert first.index("RESP-A") < first.index("RESP-B")
    assert second.index("RESP-B") < second.index("RESP-A")


async def test_pairwise_disagreement_yields_tie() -> None:
    judge = PairwiseJudge(criterion=PAIR_CRITERION, judge_model="fake/judge")
    # Call 1 says A; call 2 (swapped) also says A => un-swapped B => disagree.
    client = FakeLLMClient(script=[_verdict("A"), _verdict("A")])
    verdict = await judge.compare(RECORD, "resp-a", "resp-b", client)
    assert verdict.winner == "TIE"


async def test_pairwise_tie_reasoning_contains_both_orderings() -> None:
    """On position-swap disagreement, reasoning embeds both orderings' text."""
    judge = PairwiseJudge(criterion=PAIR_CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(
        script=[
            _verdict("A", reasoning="first reasoning"),
            _verdict("A", reasoning="second reasoning"),
        ]
    )
    verdict = await judge.compare(RECORD, "resp-a", "resp-b", client)
    assert verdict.winner == "TIE"
    assert "Position-swap disagreement" in verdict.reasoning
    assert "[A-order]" in verdict.reasoning
    assert "[B-order]" in verdict.reasoning
    assert "first reasoning" in verdict.reasoning
    assert "second reasoning" in verdict.reasoning


async def test_pairwise_tie_agreement() -> None:
    judge = PairwiseJudge(criterion=PAIR_CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_verdict("TIE"), _verdict("tie")])
    verdict = await judge.compare(RECORD, "resp-a", "resp-b", client)
    assert verdict.winner == "TIE"


async def test_pairwise_robust_verdict_parsing() -> None:
    judge = PairwiseJudge(criterion=PAIR_CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(
        script=[
            _verdict("I think response A is better."),
            _verdict("Response B is the stronger one."),
        ]
    )
    verdict = await judge.compare(RECORD, "resp-a", "resp-b", client)
    assert verdict.winner == "A"


async def test_pairwise_unparseable_verdict_raises() -> None:
    judge = PairwiseJudge(criterion=PAIR_CRITERION, judge_model="fake/judge")
    client = FakeLLMClient(script=[_verdict("no verdict here"), _verdict("still none")])
    with pytest.raises(JudgeError):
        await judge.compare(RECORD, "resp-a", "resp-b", client)
