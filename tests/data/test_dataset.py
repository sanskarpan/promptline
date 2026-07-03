"""Tests for promptline.data.dataset (Task 13)."""

from __future__ import annotations

import pathlib

import pytest

from promptline.data.dataset import (
    Dataset,
    Record,
    Turn,
    contamination_check,
    content_hash,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_record(
    prompt: str = "Hello",
    response: str = "World",
    human_label: float | None = None,
) -> Record:
    return Record(
        conversation=[Turn(role="user", content=prompt)],
        reference_output=response,
        human_label=human_label,
        meta={"source": "test"},
    )


SAMPLE_RECORDS = [
    make_record("What is 2+2?", "4", human_label=1.0),
    make_record("Name the planets.", "Mercury Venus Earth…", human_label=0.8),
    make_record("Translate 'hello' to French.", "Bonjour"),
]


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------


def test_jsonl_roundtrip(tmp_path: pathlib.Path) -> None:
    ds = Dataset(SAMPLE_RECORDS)
    fpath = tmp_path / "records.jsonl"
    ds.to_jsonl(fpath)
    ds2 = Dataset.from_jsonl(fpath)
    assert len(ds2) == len(ds)
    for orig, loaded in zip(ds, ds2):
        assert orig == loaded


def test_jsonl_roundtrip_with_dict_label(tmp_path: pathlib.Path) -> None:
    record = Record(
        conversation=[Turn(role="user", content="Rate this")],
        human_label={"quality": 0.9, "safety": 1.0},
    )
    ds = Dataset([record])
    fpath = tmp_path / "out.jsonl"
    ds.to_jsonl(fpath)
    ds2 = Dataset.from_jsonl(fpath)
    assert ds2[0] == record


# ---------------------------------------------------------------------------
# to_examples
# ---------------------------------------------------------------------------


def test_to_examples_structure() -> None:
    record = Record(
        conversation=[
            Turn(role="user", content="Hi"),
            Turn(role="assistant", content="Hello!"),
        ],
        reference_output="Hello!",
        human_label=0.95,
        meta={"domain": "chat"},
    )
    ds = Dataset([record])
    examples = ds.to_examples()
    assert len(examples) == 1
    ex = examples[0]
    assert ex.inputs["conversation"] == "user: Hi\nassistant: Hello!"
    assert ex.labels == {"reference": "Hello!"}
    assert ex.meta["human_label"] == 0.95
    assert ex.meta["domain"] == "chat"


def test_to_examples_no_reference() -> None:
    record = Record(conversation=[Turn(role="user", content="Ping")])
    ex = Dataset([record]).to_examples()[0]
    assert ex.labels == {}


def test_to_examples_no_human_label() -> None:
    record = Record(
        conversation=[Turn(role="user", content="Ping")],
        reference_output="Pong",
    )
    ex = Dataset([record]).to_examples()[0]
    assert "human_label" not in ex.meta


# ---------------------------------------------------------------------------
# Sequence interface
# ---------------------------------------------------------------------------


def test_len() -> None:
    assert len(Dataset(SAMPLE_RECORDS)) == 3


def test_iter() -> None:
    ds = Dataset(SAMPLE_RECORDS)
    assert list(ds) == SAMPLE_RECORDS


def test_indexing() -> None:
    ds = Dataset(SAMPLE_RECORDS)
    assert ds[0] == SAMPLE_RECORDS[0]
    assert ds[-1] == SAMPLE_RECORDS[-1]
    assert ds[0:2] == SAMPLE_RECORDS[0:2]


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------


def test_content_hash_is_hex_string() -> None:
    h = content_hash(SAMPLE_RECORDS[0])
    assert isinstance(h, str)
    assert len(h) == 64
    int(h, 16)  # must be valid hex


def test_content_hash_same_record_twice() -> None:
    r = make_record("stable", "content")
    assert content_hash(r) == content_hash(r)


def test_content_hash_different_records() -> None:
    r1 = make_record("a", "b")
    r2 = make_record("c", "d")
    assert content_hash(r1) != content_hash(r2)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def _make_many_records(n: int) -> list[Record]:
    return [make_record(f"prompt {i}", f"answer {i}") for i in range(n)]


def test_split_deterministic_same_seed() -> None:
    ds = Dataset(_make_many_records(50))
    s1 = ds.split({"train": 0.8, "test": 0.2}, seed=42)
    s2 = ds.split({"train": 0.8, "test": 0.2}, seed=42)
    assert [r for r in s1["train"]] == [r for r in s2["train"]]
    assert [r for r in s1["test"]] == [r for r in s2["test"]]


def test_split_different_seeds_differ() -> None:
    ds = Dataset(_make_many_records(50))
    s1 = ds.split({"train": 0.8, "test": 0.2}, seed=0)
    s2 = ds.split({"train": 0.8, "test": 0.2}, seed=99)
    # With 50 records and different seeds, assignment should differ for some
    assert [r for r in s1["train"]] != [r for r in s2["train"]]


def test_split_disjoint() -> None:
    ds = Dataset(_make_many_records(60))
    splits = ds.split({"train": 0.7, "val": 0.1, "test": 0.2}, seed=7)
    hashes_train = {content_hash(r) for r in splits["train"]}
    hashes_val = {content_hash(r) for r in splits["val"]}
    hashes_test = {content_hash(r) for r in splits["test"]}
    assert hashes_train.isdisjoint(hashes_val)
    assert hashes_train.isdisjoint(hashes_test)
    assert hashes_val.isdisjoint(hashes_test)


def test_split_covers_all_records() -> None:
    n = 80
    ds = Dataset(_make_many_records(n))
    splits = ds.split({"a": 0.5, "b": 0.5}, seed=0)
    total = sum(len(v) for v in splits.values())
    assert total == n


def test_split_roughly_proportional() -> None:
    n = 200
    ds = Dataset(_make_many_records(n))
    splits = ds.split({"a": 0.5, "b": 0.5}, seed=1)
    assert 60 <= len(splits["a"]) <= 140
    assert 60 <= len(splits["b"]) <= 140


def test_split_reorder_same_assignment() -> None:
    """Hash-based split: shuffling the dataset must not change per-record assignment."""
    records = _make_many_records(30)
    ds_orig = Dataset(records)
    ds_shuffled = Dataset(records[::-1])
    orig_splits = ds_orig.split({"x": 0.6, "y": 0.4}, seed=3)
    shuf_splits = ds_shuffled.split({"x": 0.6, "y": 0.4}, seed=3)
    orig_hashes_x = {content_hash(r) for r in orig_splits["x"]}
    shuf_hashes_x = {content_hash(r) for r in shuf_splits["x"]}
    assert orig_hashes_x == shuf_hashes_x


def test_split_bad_fractions_raises() -> None:
    ds = Dataset(_make_many_records(10))
    with pytest.raises(ValueError, match="sum"):
        ds.split({"a": 0.5, "b": 0.4})


def test_split_bad_fractions_over_one() -> None:
    ds = Dataset(_make_many_records(10))
    with pytest.raises(ValueError):
        ds.split({"a": 0.6, "b": 0.6})


# ---------------------------------------------------------------------------
# contamination_check
# ---------------------------------------------------------------------------


def test_contamination_detected() -> None:
    shared = make_record("shared", "response")
    ds_a = Dataset([make_record("unique_a"), shared])
    ds_b = Dataset([make_record("unique_b"), shared])
    overlaps = contamination_check(ds_a, ds_b)
    assert len(overlaps) == 1
    assert overlaps[0] == content_hash(shared)


def test_contamination_empty_when_disjoint() -> None:
    ds_a = Dataset([make_record("alpha"), make_record("beta")])
    ds_b = Dataset([make_record("gamma"), make_record("delta")])
    assert contamination_check(ds_a, ds_b) == []


def test_contamination_multiple_shared() -> None:
    r1 = make_record("shared1", "resp1")
    r2 = make_record("shared2", "resp2")
    ds_a = Dataset([r1, r2, make_record("only_a")])
    ds_b = Dataset([r1, r2, make_record("only_b")])
    overlaps = contamination_check(ds_a, ds_b)
    assert set(overlaps) == {content_hash(r1), content_hash(r2)}
