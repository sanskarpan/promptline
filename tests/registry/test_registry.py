from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from promptline.core.types import Candidate, ModuleState
from promptline.registry.registry import PromptRegistry


def _candidate(
    instruction: str = "Answer.", parent_ids: list[str] | None = None
) -> Candidate:
    return Candidate(
        id=f"cand-{instruction}",
        modules={"main": ModuleState(instruction=instruction)},
        parent_ids=parent_ids or [],
    )


@pytest.fixture()
def registry(tmp_path: Path) -> PromptRegistry:
    return PromptRegistry(tmp_path / ".promptline")


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


def test_register_get_round_trip(registry: PromptRegistry) -> None:
    cand = _candidate("hello")
    returned_id = registry.register(cand, program="main", run_id="run-1")
    assert returned_id == cand.id
    loaded = registry.get(cand.id)
    assert loaded == cand


def test_get_missing_returns_none(registry: PromptRegistry) -> None:
    assert registry.get("nope") is None


def test_register_idempotent(registry: PromptRegistry) -> None:
    cand = _candidate("hello")
    registry.register(cand, program="main")
    registry.register(cand, program="main")
    prompts = registry.list_prompts("main")
    assert len(prompts) == 1
    assert prompts[0]["id"] == cand.id


def test_registry_creates_parent_dirs(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / ".promptline"
    PromptRegistry(deep)
    assert (deep / "registry.db").exists()


# ---------------------------------------------------------------------------
# activate / get_active
# ---------------------------------------------------------------------------


def test_activate_unknown_prompt_raises(registry: PromptRegistry) -> None:
    with pytest.raises(KeyError):
        registry.activate("main", "unknown-id")


def test_activate_then_get_active(registry: PromptRegistry) -> None:
    cand = _candidate("hello")
    registry.register(cand, program="main")
    registry.activate("main", cand.id)
    active = registry.get_active("main")
    assert active is not None
    active_id, active_cand = active
    assert active_id == cand.id
    assert active_cand == cand


def test_get_active_none_when_never_activated(registry: PromptRegistry) -> None:
    assert registry.get_active("main") is None


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


def test_rollback_reverts_to_previous_distinct(registry: PromptRegistry) -> None:
    a, b = _candidate("A"), _candidate("B")
    registry.register(a, program="main")
    registry.register(b, program="main")
    registry.activate("main", a.id)
    registry.activate("main", b.id)
    reverted = registry.rollback("main")
    assert reverted == a.id
    active = registry.get_active("main")
    assert active is not None and active[0] == a.id


def test_rollback_twice_raises(registry: PromptRegistry) -> None:
    a, b = _candidate("A"), _candidate("B")
    registry.register(a, program="main")
    registry.register(b, program="main")
    registry.activate("main", a.id)
    registry.activate("main", b.id)
    registry.rollback("main")
    with pytest.raises(RuntimeError, match="no previous activation"):
        registry.rollback("main")


def test_rollback_without_history_raises(registry: PromptRegistry) -> None:
    with pytest.raises(RuntimeError, match="no previous activation"):
        registry.rollback("main")


def test_history_rows_appended(registry: PromptRegistry, tmp_path: Path) -> None:
    a, b = _candidate("A"), _candidate("B")
    registry.register(a, program="main")
    registry.register(b, program="main")
    registry.activate("main", a.id)
    registry.activate("main", b.id)
    registry.rollback("main")

    conn = sqlite3.connect(str(tmp_path / ".promptline" / "registry.db"))
    rows = conn.execute(
        "SELECT prompt_id, action FROM activation_history "
        "WHERE program = ? ORDER BY id",
        ("main",),
    ).fetchall()
    conn.close()
    assert rows == [
        (a.id, "activate"),
        (b.id, "activate"),
        (a.id, "rollback"),
    ]


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------


def test_lineage_diamond(registry: PromptRegistry) -> None:
    a = _candidate("A")
    b = _candidate("B", parent_ids=[a.id])
    c = _candidate("C", parent_ids=[a.id])
    d = _candidate("D", parent_ids=[b.id, c.id])
    for cand in (a, b, c, d):
        registry.register(cand, program="main")
    assert registry.lineage(d.id) == [b.id, c.id, a.id]


def test_lineage_of_seed_is_empty(registry: PromptRegistry) -> None:
    a = _candidate("A")
    registry.register(a, program="main")
    assert registry.lineage(a.id) == []


# ---------------------------------------------------------------------------
# evals
# ---------------------------------------------------------------------------


def test_evals_append_only_latest_in_list_prompts(registry: PromptRegistry) -> None:
    cand = _candidate("hello")
    registry.register(cand, program="main", run_id="run-1")
    registry.record_eval(cand.id, dataset_hash="h1", mean_score=0.5, n=50)
    registry.record_eval(cand.id, dataset_hash="h1", mean_score=0.7, n=50)
    prompts = registry.list_prompts("main")
    assert len(prompts) == 1
    row = prompts[0]
    assert row["id"] == cand.id
    assert row["run_id"] == "run-1"
    assert row["mean_score"] == 0.7


def test_list_prompts_without_evals(registry: PromptRegistry) -> None:
    cand = _candidate("hello")
    registry.register(cand, program="main")
    prompts = registry.list_prompts("main")
    assert prompts[0]["mean_score"] is None


def test_list_prompts_filters_by_program(registry: PromptRegistry) -> None:
    a, b = _candidate("A"), _candidate("B")
    registry.register(a, program="p1")
    registry.register(b, program="p2")
    ids = [row["id"] for row in registry.list_prompts("p1")]
    assert ids == [a.id]


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_persistence_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / ".promptline"
    reg1 = PromptRegistry(path)
    cand = _candidate("hello")
    reg1.register(cand, program="main")
    reg1.activate("main", cand.id)

    reg2 = PromptRegistry(path)
    assert reg2.get(cand.id) == cand
    active = reg2.get_active("main")
    assert active is not None and active[0] == cand.id
