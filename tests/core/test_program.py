from __future__ import annotations

import pytest

from promptline.core.llm import FakeLLMClient, LLMResponse
from promptline.core.program import ModelConfig, Module, Prediction, PromptProgram
from promptline.core.types import Candidate, Demo, Example, Field, ModuleState, Signature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(model: str = "test-model") -> ModelConfig:
    return ModelConfig(task_model=model, temperature=0.0, max_tokens=256)


def _single_program() -> PromptProgram:
    """One module: question → answer."""
    sig = Signature(
        instruction="Answer the question.",
        inputs=[Field("question")],
        outputs=[Field("answer")],
    )
    return PromptProgram(modules=[Module(name="main", signature=sig)])


def _multi_output_program() -> PromptProgram:
    """One module with two outputs so parse_output returns None for plain text."""
    sig = Signature(
        instruction="Answer both questions.",
        inputs=[Field("question")],
        outputs=[Field("answer"), Field("reasoning")],
    )
    return PromptProgram(modules=[Module(name="main", signature=sig)])


def _multi_output_candidate(instruction: str = "Answer both questions.") -> Candidate:
    return Candidate.seed(modules={"main": ModuleState(instruction=instruction)})


def _single_candidate(
    instruction: str = "Answer the question.", demos: list[Demo] | None = None
) -> Candidate:
    return Candidate.seed(
        modules={
            "main": ModuleState(instruction=instruction, demos=demos or []),
        }
    )


def _example(question: str = "What is 2+2?") -> Example:
    return Example(inputs={"question": question})


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


def test_model_config_defaults() -> None:
    cfg = ModelConfig(task_model="gpt-4o")
    assert cfg.reflection_model == ""
    assert cfg.judge_model == ""
    assert cfg.temperature == pytest.approx(0.2)
    assert cfg.max_tokens == 1024


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_prediction_failure_classmethod() -> None:
    pred = Prediction.failure("oops", [], 0.5)
    assert pred.failed is True
    assert pred.failure_reason == "oops"
    assert pred.outputs == {}
    assert pred.cost_usd == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# PromptProgram construction
# ---------------------------------------------------------------------------


def test_module_names_preserves_order() -> None:
    prog = PromptProgram(
        modules=[
            Module(name="a", signature=Signature("x", [], [])),
            Module(name="b", signature=Signature("y", [], [])),
        ]
    )
    assert prog.module_names == ["a", "b"]


def test_simple_classmethod() -> None:
    prog = PromptProgram.simple(
        instruction="Do the thing.",
        inputs=["x", "y"],
        outputs=["z"],
        name="step",
    )
    assert prog.module_names == ["step"]
    mod = prog.modules[0]
    assert mod.signature.instruction == "Do the thing."
    assert [f.name for f in mod.signature.inputs] == ["x", "y"]
    assert [f.name for f in mod.signature.outputs] == ["z"]


# ---------------------------------------------------------------------------
# run() — single-module happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_module_happy_path() -> None:
    fake = FakeLLMClient(script=["[[answer]]: 4"])
    prog = _single_program()
    pred = await prog.run(_example(), _single_candidate(), fake, _cfg())

    assert pred.failed is False
    assert pred.outputs == {"answer": "4"}
    assert len(pred.traces) == 1
    assert isinstance(pred.cost_usd, float)


@pytest.mark.asyncio
async def test_trace_contains_expected_fields() -> None:
    fake = FakeLLMClient(script=["[[answer]]: 4"])
    prog = _single_program()
    pred = await prog.run(_example(), _single_candidate(), fake, _cfg())

    trace = pred.traces[0]
    assert trace.module == "main"
    assert "Answer the question." in trace.system_prompt
    assert "question" in trace.user_prompt
    assert trace.raw_output == "[[answer]]: 4"
    assert trace.parsed == {"answer": "4"}


# ---------------------------------------------------------------------------
# run() — demos rendered as alternating user/assistant turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_demos_rendered_as_alternating_turns() -> None:
    demos = [
        Demo(inputs={"question": "1+1?"}, outputs={"answer": "2"}),
        Demo(inputs={"question": "3+3?"}, outputs={"answer": "6"}),
    ]
    fake = FakeLLMClient(script=["[[answer]]: 4"])
    prog = _single_program()
    candidate = _single_candidate(demos=demos)
    pred = await prog.run(_example("2+2?"), candidate, fake, _cfg())

    assert pred.failed is False
    # Inspect the messages sent to the LLM
    messages = list(fake.calls[0].messages)
    # system + 2*(user+assistant) + real user = 6 messages
    assert len(messages) == 6
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert "1+1?" in messages[1].content
    assert messages[2].role == "assistant"
    assert "[[answer]]" in messages[2].content
    assert "2" in messages[2].content
    assert messages[3].role == "user"
    assert "3+3?" in messages[3].content
    assert messages[4].role == "assistant"
    assert "6" in messages[4].content
    # Real input
    assert messages[5].role == "user"
    assert "2+2?" in messages[5].content


# ---------------------------------------------------------------------------
# run() — repair path: first reply malformed, second good
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_path_success() -> None:
    # Multi-output signature so plain text → parse_output returns None.
    good = "[[answer]]: 42\n[[reasoning]]: simple math"
    fake = FakeLLMClient(script=["This is not formatted correctly.", good])
    prog = _multi_output_program()
    pred = await prog.run(_example(), _multi_output_candidate(), fake, _cfg())

    assert pred.failed is False
    assert pred.outputs == {"answer": "42", "reasoning": "simple math"}
    # One trace for initial call + one trace for repair
    assert len(pred.traces) == 2
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_repair_messages_include_original_and_prompt() -> None:
    good = "[[answer]]: 99\n[[reasoning]]: trivial"
    fake = FakeLLMClient(script=["bad", good])
    prog = _multi_output_program()
    await prog.run(_example(), _multi_output_candidate(), fake, _cfg())

    repair_messages = list(fake.calls[1].messages)
    # Last two messages should be: assistant (bad output), user (repair prompt)
    assert repair_messages[-2].role == "assistant"
    assert repair_messages[-2].content == "bad"
    assert repair_messages[-1].role == "user"
    assert "required format" in repair_messages[-1].content.lower()


# ---------------------------------------------------------------------------
# run() — repair fails twice → failed Prediction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_fails_returns_failed_prediction() -> None:
    fake = FakeLLMClient(script=["bad output", "still bad"])
    prog = _multi_output_program()
    pred = await prog.run(_example(), _multi_output_candidate(), fake, _cfg())

    assert pred.failed is True
    assert "main" in pred.failure_reason
    assert "unparseable" in pred.failure_reason
    # Two traces: initial + repair attempt
    assert len(pred.traces) == 2


# ---------------------------------------------------------------------------
# run() — two-module program: module2 sees module1 outputs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_module_program_pipes_outputs() -> None:
    sig1 = Signature(
        instruction="Extract city.",
        inputs=[Field("query")],
        outputs=[Field("city")],
    )
    sig2 = Signature(
        instruction="Name the country for this city.",
        inputs=[Field("city")],
        outputs=[Field("country")],
    )
    prog = PromptProgram(
        modules=[
            Module(name="extract", signature=sig1),
            Module(name="lookup", signature=sig2),
        ]
    )
    candidate = Candidate.seed(
        modules={
            "extract": ModuleState(instruction="Extract city."),
            "lookup": ModuleState(instruction="Name the country for this city."),
        }
    )
    example = Example(inputs={"query": "Where is Paris?"})

    fake = FakeLLMClient(
        script=[
            "[[city]]: Paris",
            "[[country]]: France",
        ]
    )
    pred = await prog.run(example, candidate, fake, _cfg())

    assert pred.failed is False
    assert pred.outputs.get("city") == "Paris"
    assert pred.outputs.get("country") == "France"

    # Second call's user content must include module1's output value
    second_call_messages = list(fake.calls[1].messages)
    user_messages = [m for m in second_call_messages if m.role == "user"]
    assert any("Paris" in m.content for m in user_messages), (
        f"Expected 'Paris' in second module's user messages, got: {user_messages}"
    )


@pytest.mark.asyncio
async def test_two_module_cost_accumulated() -> None:
    """Total cost is the sum across all LLM calls."""

    class _CostFake:
        def __init__(self, costs: list[float]) -> None:
            self._costs = list(costs)
            self.calls: list = []

        async def complete(self, call):
            self.calls.append(call)
            return LLMResponse(text="[[out]]: x", cost_usd=self._costs.pop(0))

    sig = Signature("Do.", inputs=[Field("inp")], outputs=[Field("out")])
    prog = PromptProgram(
        modules=[
            Module(name="a", signature=sig),
            Module(
                name="b", signature=Signature("Do.", inputs=[Field("out")], outputs=[Field("out")])
            ),
        ]
    )
    candidate = Candidate.seed(
        modules={
            "a": ModuleState(instruction="Do."),
            "b": ModuleState(instruction="Do."),
        }
    )
    example = Example(inputs={"inp": "hello"})

    fake = _CostFake([0.01, 0.02])
    pred = await prog.run(example, candidate, fake, _cfg())

    assert pred.cost_usd == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# run() — module whose declared inputs are never produced → wiring failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_module_with_no_present_inputs_fails_without_llm_call() -> None:
    """Module 2's declared input is never produced → Prediction.failed, no doomed LLM call."""
    sig1 = Signature(
        instruction="Extract city.",
        inputs=[Field("query")],
        outputs=[Field("city")],
    )
    sig2 = Signature(
        instruction="Summarise the topic.",
        inputs=[Field("topic")],  # never produced by module 1
        outputs=[Field("summary")],
    )
    prog = PromptProgram(
        modules=[
            Module(name="extract", signature=sig1),
            Module(name="summarise", signature=sig2),
        ]
    )
    candidate = Candidate.seed(
        modules={
            "extract": ModuleState(instruction="Extract city."),
            "summarise": ModuleState(instruction="Summarise the topic."),
        }
    )
    example = Example(inputs={"query": "Where is Paris?"})

    # Only one scripted reply: module 2 must NOT reach the LLM.
    fake = FakeLLMClient(script=["[[city]]: Paris"])
    pred = await prog.run(example, candidate, fake, _cfg())

    assert pred.failed is True
    assert "summarise" in pred.failure_reason
    assert "no declared inputs" in pred.failure_reason
    # Module 1 called the LLM once; module 2 did not.
    assert len(fake.calls) == 1
