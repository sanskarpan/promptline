"""Tests for the ProTeGi optimizer (textual gradients + racing)."""
from __future__ import annotations

from promptline.core.llm import FakeLLMClient, LLMCall
from promptline.core.program import ModelConfig, Prediction, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, MetricResult
from promptline.optimizers.base import RunEvent
from promptline.optimizers.protegi import ProTeGi

MARKER = "ALWAYS CITE"
FEEDBACK = "missing citation of sources"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _program() -> PromptProgram:
    return PromptProgram.simple(
        instruction="Answer the question.",
        inputs=["question"],
        outputs=["answer"],
    )


def _seed(program: PromptProgram) -> Candidate:
    return Candidate.seed(
        modules={
            m.name: ModuleState(instruction=m.signature.instruction)
            for m in program.modules
        }
    )


def _trainset(n: int = 20) -> list[Example]:
    return [Example(inputs={"question": f"Q{i}?"}) for i in range(n)]


def _marker_metric(example: Example, prediction: Prediction) -> MetricResult:
    if MARKER in prediction.outputs.get("answer", ""):
        return MetricResult(score=1.0, feedback="cited sources")
    return MetricResult(score=0.0, feedback=FEEDBACK)


def _client(edited_instruction: str) -> FakeLLMClient:
    """Scripted client covering all four call types.

    - Gradient calls (diagnose): return a short critique.
    - Edit calls (rewrite): return *edited_instruction* in a fenced block.
    - Paraphrase calls: return a paraphrase preserving the marker iff the
      instruction being paraphrased contains it.
    - Task calls: emit the marker iff the system prompt contains it.
    """

    def _respond(call: LLMCall) -> str:
        joined = "\n".join(m.content for m in call.messages)
        if "diagnose why this instruction failed" in joined:
            return "The instruction never asks the model to cite sources."
        if "Rewrite the instruction" in joined:
            return f"Here you go.\n```\n{edited_instruction}\n```"
        if "Paraphrase this instruction" in joined:
            if MARKER in joined:
                return f"```\nRespond to the question. {MARKER} in every answer.\n```"
            return "```\nRespond to the question.\n```"
        system = call.messages[0].content
        if MARKER in system:
            return f"[[answer]]: sources cited. {MARKER}"
        return "[[answer]]: plain answer"

    return FakeLLMClient(script=_respond)


def _harness(client: FakeLLMClient) -> EvalHarness:
    return EvalHarness(client, ModelConfig(task_model="fake"), concurrency=4)


async def _run(
    client: FakeLLMClient,
    trainset: list[Example] | None = None,
    budget: Budget | None = None,
    events: list[RunEvent] | None = None,
    **params: object,
):
    program = _program()
    seed = _seed(program)
    optimizer = ProTeGi(**params)  # type: ignore[arg-type]
    result = await optimizer.optimize(
        program,
        seed,
        trainset if trainset is not None else _trainset(),
        _marker_metric,
        budget if budget is not None else Budget(max_rollouts=1000),
        _harness(client),
        emit=events.append if events is not None else (lambda e: None),
    )
    return result, seed


# ---------------------------------------------------------------------------
# Gradient prompt content
# ---------------------------------------------------------------------------


async def test_failing_examples_appear_in_gradient_prompt() -> None:
    client = _client(f"Answer the question. {MARKER} sources.")
    await _run(client, n_rounds=1)

    gradient_calls = [
        c
        for c in client.calls
        if "diagnose why this instruction failed" in "\n".join(m.content for m in c.messages)
    ]
    assert gradient_calls, "no gradient calls were made"
    joined = "\n".join(m.content for m in gradient_calls[0].messages)
    assert FEEDBACK in joined  # failure feedback surfaced
    assert "Q" in joined  # example inputs excerpt surfaced


# ---------------------------------------------------------------------------
# Gradient -> edit -> child pipeline
# ---------------------------------------------------------------------------


async def test_edit_child_wins_racing() -> None:
    client = _client(f"Answer the question. {MARKER} sources.")
    result, seed = await _run(client)

    assert MARKER in result.best.modules["main"].instruction
    assert result.best.id != seed.id
    assert result.scores[result.best.id] > result.scores[seed.id]


# ---------------------------------------------------------------------------
# Paraphrase children
# ---------------------------------------------------------------------------


async def test_paraphrase_children_created() -> None:
    client = _client(f"Answer the question. {MARKER} sources.")
    events: list[RunEvent] = []
    await _run(client, events=events, n_rounds=1, n_gradients=2, n_paraphrases=1)

    proposed = [e for e in events if e.type == "candidate_proposed"]
    para = [e for e in proposed if e.payload.get("source") == "paraphrase"]
    grad = [e for e in proposed if e.payload.get("source") == "gradient"]
    assert len(grad) == 2  # n_gradients edits from the single seed parent
    assert len(para) == 2  # one paraphrase per edited child


# ---------------------------------------------------------------------------
# Racing
# ---------------------------------------------------------------------------


async def test_racing_drops_losers() -> None:
    client = _client(f"Answer the question. {MARKER} sources.")
    events: list[RunEvent] = []
    await _run(
        client,
        events=events,
        n_rounds=1,
        beam_width=2,
        n_gradients=2,
        n_paraphrases=1,
        racing_rounds=3,
    )

    racing = [
        e
        for e in events
        if e.type == "minibatch_scored" and e.payload.get("phase") == "racing"
    ]
    assert racing, "no racing evaluations happened"
    rounds = sorted({e.payload["racing_round"] for e in racing})
    assert len(rounds) >= 2, "racing should have run multiple halving rounds"
    ids_by_round = {
        r: {e.payload["candidate_id"] for e in racing if e.payload["racing_round"] == r}
        for r in rounds
    }
    first, later = ids_by_round[rounds[0]], ids_by_round[rounds[-1]]
    assert later < first  # strict subset: some candidates were dropped
    assert len(later) >= 2  # never fewer than beam_width survivors raced


async def test_racing_cheaper_than_full_eval() -> None:
    client = _client(f"Answer the question. {MARKER} sources.")
    trainset = _trainset(50)
    budget = Budget(max_rollouts=10_000)
    n_rounds = 2
    result, _ = await _run(client, trainset=trainset, budget=budget, n_rounds=n_rounds)

    pool_size = len(result.candidates)
    assert pool_size > 4  # a real pool formed
    # Racing (batch-based selection) + final full-eval on the beam must be
    # cheaper than naively full-evaluating every unique candidate on the full
    # trainset at each optimization round.
    assert budget.rollouts_used < pool_size * len(trainset) * n_rounds


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


async def test_budget_early_stop() -> None:
    client = _client(f"Answer the question. {MARKER} sources.")
    budget = Budget(max_rollouts=5)
    result, seed = await _run(client, budget=budget)

    assert budget.rollouts_used <= 5
    assert result.best is not None
    assert seed.id in result.scores


# ---------------------------------------------------------------------------
# Events + determinism
# ---------------------------------------------------------------------------


async def test_event_sequence_sane() -> None:
    client = _client(f"Answer the question. {MARKER} sources.")
    events: list[RunEvent] = []
    result, _ = await _run(client, events=events, n_rounds=2)

    types = [e.type for e in events]
    assert types[0] == "run_started"
    assert types[-1] == "run_finished"
    assert types.count("budget_tick") == 2  # one per round
    assert "candidate_proposed" in types
    assert "minibatch_scored" in types
    # every proposed candidate carries a source tag
    for e in events:
        if e.type == "candidate_proposed":
            assert e.payload["source"] in ("gradient", "paraphrase")
    assert result.events_count == len(events)


async def test_determinism_same_seed_same_best() -> None:
    edited = f"Answer the question. {MARKER} sources."
    r1, _ = await _run(_client(edited), rng_seed=7)
    r2, _ = await _run(_client(edited), rng_seed=7)
    assert (
        r1.best.modules["main"].instruction == r2.best.modules["main"].instruction
    )


# ---------------------------------------------------------------------------
# Full-eval
# ---------------------------------------------------------------------------


async def test_full_eval_events_emitted() -> None:
    """full_eval events must be emitted for each final beam candidate."""
    client = _client(f"Answer the question. {MARKER} sources.")
    events: list[RunEvent] = []
    result, _ = await _run(client, events=events, n_rounds=1, beam_width=2)

    full_eval_events = [e for e in events if e.type == "full_eval"]
    assert full_eval_events, "no full_eval events emitted"
    # At most beam_width events (one per beam candidate).
    assert len(full_eval_events) <= 2
    for e in full_eval_events:
        assert "candidate_id" in e.payload
        assert "mean_score" in e.payload
        assert "n" in e.payload
        assert "truncated" in e.payload
    # full_eval events appear before run_finished.
    types = [e.type for e in events]
    last_full_eval_idx = max(i for i, t in enumerate(types) if t == "full_eval")
    run_finished_idx = types.index("run_finished")
    assert last_full_eval_idx < run_finished_idx


async def test_best_chosen_by_full_eval_score() -> None:
    """result.best must be the beam candidate with the highest full-eval mean."""
    client = _client(f"Answer the question. {MARKER} sources.")
    events: list[RunEvent] = []
    result, _ = await _run(client, events=events, n_rounds=1)

    non_truncated = [
        e for e in events if e.type == "full_eval" and not e.payload.get("truncated")
    ]
    assert non_truncated, "expected at least one non-truncated full_eval event"
    best_event = max(non_truncated, key=lambda e: e.payload["mean_score"])
    assert result.best.id == best_event.payload["candidate_id"]


async def test_full_eval_budget_exhaustion_returns_cleanly() -> None:
    """Optimizer must return cleanly and emit full_eval events even when budget
    exhausts mid-full-eval (some candidates get truncated=True events)."""
    client = _client(f"Answer the question. {MARKER} sources.")
    events: list[RunEvent] = []
    # Budget sized so the main loop completes but full-eval is at least partially
    # cut short (minibatch=8 + some racing rollouts ≤ 20 < full trainset × beam).
    budget = Budget(max_rollouts=20)
    result, _ = await _run(client, events=events, budget=budget, n_rounds=1)

    assert result.best is not None
    full_eval_events = [e for e in events if e.type == "full_eval"]
    # At least one full_eval event must still be emitted (truncated or not).
    assert full_eval_events, "full_eval events must be emitted even with exhausted budget"
    for e in full_eval_events:
        assert "truncated" in e.payload


async def test_unevaluated_candidate_not_in_scores() -> None:
    """Candidates never scored (racing or minibatch) must be absent from scores."""
    client = _client(f"Answer the question. {MARKER} sources.")
    # Budget tight enough that seed minibatch exhausts almost all rollouts:
    # seed minibatch uses 8, leaving 7 for racing → only seed gets racing-evaluated,
    # all children remain unscored.  n_gradients=2,n_paraphrases=1 yields 4 children.
    budget = Budget(max_rollouts=15)
    result, seed = await _run(
        client,
        budget=budget,
        n_rounds=1,
        n_gradients=2,
        n_paraphrases=1,
    )

    # Seed is always minibatch-evaluated and must appear in scores.
    assert seed.id in result.scores

    # Some children should have been proposed but never evaluated.
    unscored = [c for c in result.candidates if c.id not in result.scores]
    assert unscored, "expected unevaluated children to be absent from scores"

    # The returned best must always have a score (it was evaluated somewhere).
    assert result.best.id in result.scores


# ---------------------------------------------------------------------------
# Continuous-metric failure threshold
# ---------------------------------------------------------------------------


def test_format_failures_uses_continuous_threshold() -> None:
    """Scores below failure_threshold count as failures; >= do not."""
    from promptline.core.types import Example
    from promptline.eval.harness import EvalReport, ExampleResult
    from promptline.optimizers.protegi import ProTeGi, _format_failures

    batch = [
        Example(inputs={"question": "q-low"}),
        Example(inputs={"question": "q-high"}),
    ]
    report = EvalReport(
        per_example=[
            ExampleResult(
                example_idx=0, score=0.65, feedback="meh", cost_usd=0.0,
                failed=False,
            ),
            ExampleResult(
                example_idx=1, score=0.75, feedback="good", cost_usd=0.0,
                failed=False,
            ),
        ],
    )

    assert ProTeGi().failure_threshold == 0.7  # continuous-metric default
    rendered = _format_failures(batch, report)  # default threshold 0.7
    assert "q-low" in rendered
    assert "q-high" not in rendered

    # Overridable: a stricter cutoff flags both.
    strict = _format_failures(batch, report, failure_threshold=1.0)
    assert "q-low" in strict and "q-high" in strict
