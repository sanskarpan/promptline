"""Tests for GEPA reflection prompt/parsing and system-aware merge."""
from __future__ import annotations

import random

from promptline.core.types import Candidate, ModuleState
from promptline.optimizers.gepa.merge import (
    common_ancestor,
    is_related,
    merge_candidates,
)
from promptline.optimizers.gepa.reflect import (
    DIRECTIVE,
    ReflectionExample,
    build_reflection_prompt,
    parse_new_instruction,
)

# ---------------------------------------------------------------------------
# Reflection prompt
# ---------------------------------------------------------------------------


def test_reflection_prompt_contains_everything() -> None:
    examples = [
        ReflectionExample(
            inputs="question: What is 2+2?",
            output="[[answer]]: 5",
            score=0.0,
            feedback="wrong: expected 4",
        ),
        ReflectionExample(
            inputs="question: Capital of France?",
            output="[[answer]]: Paris",
            score=1.0,
            feedback="correct",
        ),
    ]
    prompt = build_reflection_prompt("main", "Answer the question.", examples)
    assert "Answer the question." in prompt
    assert "question: What is 2+2?" in prompt
    assert "[[answer]]: 5" in prompt
    assert "Score: 0.0000" in prompt
    assert "Score: 1.0000" in prompt
    assert "wrong: expected 4" in prompt
    assert "correct" in prompt
    assert DIRECTIVE in prompt
    assert '"main"' in prompt


# ---------------------------------------------------------------------------
# Instruction parsing
# ---------------------------------------------------------------------------


def test_parse_fenced_block() -> None:
    text = "Diagnosis: bad.\n```\nNew instruction here.\n```\ntrailing"
    assert parse_new_instruction(text) == "New instruction here."


def test_parse_fenced_block_with_language_tag() -> None:
    text = "```text\nMultiline\ninstruction.\n```"
    assert parse_new_instruction(text) == "Multiline\ninstruction."


def test_parse_first_of_multiple_blocks() -> None:
    text = "```\nfirst\n```\nand\n```\nsecond\n```"
    assert parse_new_instruction(text) == "first"


def test_parse_fallback_whole_reply() -> None:
    assert parse_new_instruction("  just a bare reply  ") == "just a bare reply"


def test_parse_empty_fenced_block_falls_back() -> None:
    text = "```\n\n```"
    assert parse_new_instruction(text) == text.strip()


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _cand(instr_a: str, instr_b: str) -> Candidate:
    return Candidate.seed(
        modules={
            "m1": ModuleState(instruction=instr_a),
            "m2": ModuleState(instruction=instr_b),
        }
    )


def _diamond() -> tuple[Candidate, Candidate, Candidate, dict[str, Candidate]]:
    """Ancestor a; children b (m1 mutated) and c (m2 mutated)."""
    a = _cand("base1", "base2")
    b = a.child(
        modules={
            "m1": ModuleState(instruction="b-mut"),
            "m2": ModuleState(instruction="base2"),
        },
        optimizer="gepa",
    )
    c = a.child(
        modules={
            "m1": ModuleState(instruction="base1"),
            "m2": ModuleState(instruction="c-mut"),
        },
        optimizer="gepa",
    )
    pool = {x.id: x for x in (a, b, c)}
    return a, b, c, pool


def test_common_ancestor_diamond() -> None:
    a, b, c, pool = _diamond()
    assert common_ancestor(b.id, c.id, pool) == a.id


def test_common_ancestor_deeper_lineage() -> None:
    a, b, c, pool = _diamond()
    d = b.child(
        modules={
            "m1": ModuleState(instruction="d-mut"),
            "m2": ModuleState(instruction="base2"),
        },
        optimizer="gepa",
    )
    pool[d.id] = d
    assert common_ancestor(d.id, c.id, pool) == a.id


def test_no_common_ancestor_returns_none() -> None:
    x = _cand("x1", "x2")
    y = _cand("y1", "y2")
    pool = {x.id: x, y.id: y}
    assert common_ancestor(x.id, y.id, pool) is None


def test_is_related() -> None:
    a, b, c, pool = _diamond()
    assert is_related(a.id, b.id, pool)  # ancestor/descendant
    assert is_related(b.id, a.id, pool)
    assert is_related(a.id, a.id, pool)  # self
    assert not is_related(b.id, c.id, pool)  # siblings


# ---------------------------------------------------------------------------
# Triplet rule truth table
# ---------------------------------------------------------------------------


def test_triplet_rule_truth_table() -> None:
    rng = random.Random(0)
    ancestor = Candidate.seed(
        modules={
            "only_p1_differs": ModuleState(instruction="base"),
            "only_p2_differs": ModuleState(instruction="base"),
            "both_differ": ModuleState(instruction="base"),
            "neither_differs": ModuleState(instruction="base"),
        }
    )
    p1 = ancestor.child(
        modules={
            "only_p1_differs": ModuleState(instruction="p1"),
            "only_p2_differs": ModuleState(instruction="base"),
            "both_differ": ModuleState(instruction="p1"),
            "neither_differs": ModuleState(instruction="base"),
        },
        optimizer="gepa",
    )
    p2 = ancestor.child(
        modules={
            "only_p1_differs": ModuleState(instruction="base"),
            "only_p2_differs": ModuleState(instruction="p2"),
            "both_differ": ModuleState(instruction="p2"),
            "neither_differs": ModuleState(instruction="base"),
        },
        optimizer="gepa",
    )

    # p2 has the higher mean → wins the "both differ" module.
    child = merge_candidates(p1, p2, ancestor, mean1=0.4, mean2=0.6, rng=rng)
    assert child.modules["only_p1_differs"].instruction == "p1"
    assert child.modules["only_p2_differs"].instruction == "p2"
    assert child.modules["both_differ"].instruction == "p2"
    assert child.modules["neither_differs"].instruction == "base"
    assert child.parent_ids == [p1.id, p2.id]

    # p1 has the higher mean → wins the "both differ" module.
    child2 = merge_candidates(p1, p2, ancestor, mean1=0.9, mean2=0.1, rng=rng)
    assert child2.modules["both_differ"].instruction == "p1"


def test_triplet_rule_tie_break_is_seeded() -> None:
    ancestor = Candidate.seed(modules={"m": ModuleState(instruction="base")})
    p1 = ancestor.child(
        modules={"m": ModuleState(instruction="p1")}, optimizer="gepa"
    )
    p2 = ancestor.child(
        modules={"m": ModuleState(instruction="p2")}, optimizer="gepa"
    )
    picks = {
        merge_candidates(
            p1, p2, ancestor, 0.5, 0.5, random.Random(seed)
        ).modules["m"].instruction
        for seed in range(20)
    }
    # Tie-break picks from both parents across seeds, never the ancestor.
    assert picks == {"p1", "p2"}
