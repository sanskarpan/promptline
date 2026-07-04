"""Shared fixtures for the offline end-to-end suite.

The e2e tests compose REAL components (Calibrator, optimizers, run_gate,
PromptRegistry, FastAPI app) and replace only the LLM with a scripted
:class:`~promptline.core.llm.FakeLLMClient`.  Two scripting patterns are used:

* **Marker clients** (pattern from tests/optimizers/gepa/test_engine.py): a
  task call answers with :data:`MARKER` iff the system prompt already contains
  it, so the marker metric rewards exactly the candidates whose instruction was
  improved by the (also scripted) reflection/proposal calls.
* **Sentinel judges**: gold records embed their human label in the reference
  output (``GOLD-<label>-<i>``); the judge client parses the sentinel back out,
  which gives exact (or deliberately broken) judge/human agreement.
"""
from __future__ import annotations

import re

import pytest

from promptline.core.llm import FakeLLMClient, LLMCall
from promptline.core.program import ModelConfig, Prediction, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.data.dataset import Dataset, Record, Turn
from promptline.eval.harness import EvalHarness, MetricResult

#: Marker inserted by scripted reflection; the marker metric rewards it.
MARKER = "ALWAYS CITE SOURCES"

#: Improved instruction the scripted proposers return.
IMPROVED_INSTRUCTION = f"Answer the customer's question. {MARKER}."


# ---------------------------------------------------------------------------
# Program / trainset builders
# ---------------------------------------------------------------------------


def support_program() -> PromptProgram:
    """Single-module support-assistant style program."""
    return PromptProgram.simple(
        instruction="You are a support agent. Answer the question.",
        inputs=["conversation"],
        outputs=["answer"],
        name="support",
    )


def seed_for(program: PromptProgram) -> Candidate:
    return Candidate.seed(
        modules={
            m.name: ModuleState(instruction=m.signature.instruction)
            for m in program.modules
        }
    )


def support_trainset(n: int, prefix: str = "t") -> list[Example]:
    """Support-style examples; *prefix* keeps splits contamination-free."""
    return [
        Example(
            inputs={"conversation": f"user: {prefix}-question {i}?"},
            labels={"reference": f"{prefix}-reference {i}"},
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Marker metric + clients
# ---------------------------------------------------------------------------


def marker_metric(example: Example, prediction: Prediction) -> MetricResult:
    """Rewards outputs that contain :data:`MARKER`."""
    if MARKER in prediction.outputs.get("answer", ""):
        return MetricResult(score=1.0, feedback="cited sources")
    return MetricResult(score=0.0, feedback="did not cite sources")


def make_pipeline_client(
    improved_instruction: str = IMPROVED_INSTRUCTION,
) -> FakeLLMClient:
    """One scripted client that satisfies every optimizer's LLM call shapes.

    Task calls echo :data:`MARKER` iff the system prompt contains it.
    Proposer/reflection calls (GEPA, OPRO, ProTeGi, MIPRO) are recognised by
    their prompt boilerplate and return *improved_instruction*.
    """

    def _respond(call: LLMCall) -> str:
        blob = "\n".join(m.content for m in call.messages)
        # GEPA reflection.
        if "Diagnose the failures" in blob:
            return (
                "The answers never cite sources.\n"
                f"```\n{improved_instruction}\n```"
            )
        # OPRO trajectory proposer.
        if "<INS>" in blob:
            return f"<INS>{improved_instruction}</INS>"
        # ProTeGi textual gradient.
        if "diagnose why this instruction failed" in blob:
            return "The instruction never asks for citations."
        # MIPRO dataset summary.
        if "Summarize the patterns" in blob:
            return "- customers ask support questions\n- answers should cite sources"
        # ProTeGi edit/paraphrase + MIPRO proposals all ask for a fenced block.
        if "fenced code block" in blob:
            return f"```\n{improved_instruction}\n```"
        # Task call: marker echo.
        system = call.messages[0].content
        if MARKER in system:
            return f"[[answer]]: certainly — see docs. {MARKER}."
        return "[[answer]]: plain answer without citations"

    return FakeLLMClient(script=_respond)


def make_echo_client() -> FakeLLMClient:
    """Task client that answers with the question id (for exact-match metrics).

    The real user turn is always the LAST message; demo turns precede it.
    """

    def _respond(call: LLMCall) -> str:
        last = call.messages[-1].content
        match = re.search(r"question (\d+)", last)
        if match:
            return f"[[answer]]: echo {match.group(1)}"
        return "[[answer]]: echo ?"

    return FakeLLMClient(script=_respond)


def echo_metric(example: Example, prediction: Prediction) -> MetricResult:
    """Exact-match against the question id embedded in the conversation."""
    match = re.search(r"question (\d+)", example.inputs.get("conversation", ""))
    expected = f"echo {match.group(1)}" if match else "echo ?"
    got = prediction.outputs.get("answer", "").strip()
    ok = got == expected
    return MetricResult(score=1.0 if ok else 0.0, feedback=f"expected {expected!r}")


def make_harness(client: FakeLLMClient, concurrency: int = 4) -> EvalHarness:
    return EvalHarness(
        client,
        ModelConfig(task_model="fake/task", reflection_model="fake/reflect"),
        concurrency=concurrency,
    )


# ---------------------------------------------------------------------------
# Gold dataset + sentinel judge client
# ---------------------------------------------------------------------------

_SENTINEL_RE = re.compile(r"GOLD-(\d)-\d+")


def build_gold_dataset(n: int = 30) -> Dataset:
    """Gold records whose reference output embeds the human label (1..5)."""
    records: list[Record] = []
    for i in range(n):
        label = i % 5 + 1
        records.append(
            Record(
                conversation=[Turn(role="user", content=f"gold question {i}?")],
                reference_output=f"reply GOLD-{label}-{i} to the customer",
                human_label=float(label),
            )
        )
    return Dataset(records)


def make_judge_client(agreement: str = "high") -> FakeLLMClient:
    """Scripted judge: parses the GOLD sentinel out of the judged response.

    ``agreement="high"`` echoes the embedded human label (kappa == 1.0);
    ``agreement="low"`` always answers 3 (kappa == 0 -> cert fails);
    ``agreement="unparseable"`` returns a score field with no integer in it.
    """

    def _respond(call: LLMCall) -> str:
        blob = "\n".join(m.content for m in call.messages)
        if agreement == "unparseable":
            return "[[reasoning]]: cannot decide\n[[score]]: N/A"
        if agreement == "low":
            return "[[reasoning]]: shrug\n[[score]]: 3"
        match = _SENTINEL_RE.search(blob)
        score = match.group(1) if match else "3"
        return f"[[reasoning]]: matches the gold label\n[[score]]: {score}"

    return FakeLLMClient(script=_respond)


# ---------------------------------------------------------------------------
# Pytest fixtures (thin wrappers so tests can take them as arguments)
# ---------------------------------------------------------------------------


@pytest.fixture()
def program() -> PromptProgram:
    return support_program()


@pytest.fixture()
def seed(program: PromptProgram) -> Candidate:
    return seed_for(program)
