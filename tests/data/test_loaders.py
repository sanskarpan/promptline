"""Tests for promptline.data.loaders (Task 14).

All tests use literal fixture dicts — no network access, no ``datasets`` install
required.
"""

from __future__ import annotations

import sys

import pytest

from promptline.data.loaders import (
    _map_bitext_row,
    _map_helpsteer_row,
    _map_mtbench_row,
    load_bitext,
    load_helpsteer2,
    load_mtbench_human,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HELPSTEER_ROW = {
    "prompt": "What is the capital of France?",
    "response": "The capital of France is Paris.",
    "helpfulness": 4,
    "correctness": 4,
    "coherence": 3,
    "complexity": 2,
    "verbosity": 1,
}

BITEXT_ROW = {
    "instruction": "How do I reset my password?",
    "response": "To reset your password, click 'Forgot password' on the login page.",
    "category": "ACCOUNT",
    "intent": "reset_password",
    "flags": "BQZ",
}

MTBENCH_ROW = {
    "question_id": 101,
    "model_a": "gpt-4",
    "model_b": "claude-2",
    "winner": "model_a",
    "judge": "gpt-4",
    "turn": 2,
    "conversation_a": [
        {"role": "user", "content": "Write a short poem about the sea."},
        {"role": "assistant", "content": "The waves crash and roar,\nUpon the sunlit shore."},
        {"role": "user", "content": "Now make it rhyme better."},
        {
            "role": "assistant",
            "content": "The ocean sings a song so free,\nIts melody drifts out to sea.",
        },
    ],
    "conversation_b": [
        {"role": "user", "content": "Write a short poem about the sea."},
        {
            "role": "assistant",
            "content": "Blue water, endless sky,\nGulls and waves go sweeping by.",
        },
        {"role": "user", "content": "Now make it rhyme better."},
        {
            "role": "assistant",
            "content": "Beneath the sun's bright gleam,\nThe waters flow like a dream.",
        },
    ],
}


# ---------------------------------------------------------------------------
# _map_helpsteer_row
# ---------------------------------------------------------------------------


def test_helpsteer_conversation() -> None:
    record = _map_helpsteer_row(HELPSTEER_ROW)
    assert len(record.conversation) == 1
    turn = record.conversation[0]
    assert turn.role == "user"
    assert turn.content == HELPSTEER_ROW["prompt"]


def test_helpsteer_reference_output() -> None:
    record = _map_helpsteer_row(HELPSTEER_ROW)
    assert record.reference_output == HELPSTEER_ROW["response"]


def test_helpsteer_human_label_default_attribute() -> None:
    record = _map_helpsteer_row(HELPSTEER_ROW)
    assert record.human_label == float(HELPSTEER_ROW["helpfulness"])


def test_helpsteer_attribute_selection() -> None:
    record = _map_helpsteer_row(HELPSTEER_ROW, attribute="correctness")
    assert record.human_label == float(HELPSTEER_ROW["correctness"])
    assert record.meta["attribute"] == "correctness"


def test_helpsteer_all_ratings_in_meta() -> None:
    record = _map_helpsteer_row(HELPSTEER_ROW)
    ratings = record.meta["all_ratings"]
    for attr in ("helpfulness", "correctness", "coherence", "complexity", "verbosity"):
        assert attr in ratings
        assert ratings[attr] == HELPSTEER_ROW[attr]


def test_helpsteer_meta_attribute_key() -> None:
    record = _map_helpsteer_row(HELPSTEER_ROW)
    assert record.meta["attribute"] == "helpfulness"


# ---------------------------------------------------------------------------
# _map_bitext_row
# ---------------------------------------------------------------------------


def test_bitext_conversation() -> None:
    record = _map_bitext_row(BITEXT_ROW)
    assert len(record.conversation) == 1
    turn = record.conversation[0]
    assert turn.role == "user"
    assert turn.content == BITEXT_ROW["instruction"]


def test_bitext_reference_output() -> None:
    record = _map_bitext_row(BITEXT_ROW)
    assert record.reference_output == BITEXT_ROW["response"]


def test_bitext_meta() -> None:
    record = _map_bitext_row(BITEXT_ROW)
    assert record.meta["category"] == BITEXT_ROW["category"]
    assert record.meta["intent"] == BITEXT_ROW["intent"]
    assert record.meta["flags"] == BITEXT_ROW["flags"]


def test_bitext_no_human_label() -> None:
    record = _map_bitext_row(BITEXT_ROW)
    assert record.human_label is None


# ---------------------------------------------------------------------------
# _map_mtbench_row
# ---------------------------------------------------------------------------


def test_mtbench_user_turns_only() -> None:
    record = _map_mtbench_row(MTBENCH_ROW)
    # conversation_a has 2 user turns
    assert len(record.conversation) == 2
    for turn in record.conversation:
        assert turn.role == "user"


def test_mtbench_user_turn_content() -> None:
    record = _map_mtbench_row(MTBENCH_ROW)
    assert record.conversation[0].content == "Write a short poem about the sea."
    assert record.conversation[1].content == "Now make it rhyme better."


def test_mtbench_response_a_extracted() -> None:
    record = _map_mtbench_row(MTBENCH_ROW)
    expected = MTBENCH_ROW["conversation_a"][-1]["content"]
    assert record.meta["response_a"] == expected


def test_mtbench_response_b_extracted() -> None:
    record = _map_mtbench_row(MTBENCH_ROW)
    expected = MTBENCH_ROW["conversation_b"][-1]["content"]
    assert record.meta["response_b"] == expected


def test_mtbench_winner_in_meta() -> None:
    record = _map_mtbench_row(MTBENCH_ROW)
    assert record.meta["winner"] == "model_a"


def test_mtbench_question_id_and_judge() -> None:
    record = _map_mtbench_row(MTBENCH_ROW)
    assert record.meta["question_id"] == 101
    assert record.meta["judge"] == "gpt-4"


def test_mtbench_turn_in_meta() -> None:
    record = _map_mtbench_row(MTBENCH_ROW)
    assert record.meta["turn"] == 2


def test_mtbench_no_reference_output() -> None:
    record = _map_mtbench_row(MTBENCH_ROW)
    assert record.reference_output is None


def test_mtbench_no_human_label() -> None:
    record = _map_mtbench_row(MTBENCH_ROW)
    assert record.human_label is None


# ---------------------------------------------------------------------------
# Public loaders — ImportError when datasets is missing
# ---------------------------------------------------------------------------


def test_load_helpsteer2_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match="promptline\\[data\\]"):
        load_helpsteer2()


def test_load_bitext_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match="promptline\\[data\\]"):
        load_bitext()


def test_load_mtbench_human_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match="promptline\\[data\\]"):
        load_mtbench_human()
