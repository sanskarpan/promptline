from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from pydantic import BaseModel

from promptline.core.types import Example


class Turn(BaseModel):
    role: str
    content: str


class Record(BaseModel):
    conversation: list[Turn]
    reference_output: str | None = None
    human_label: float | dict[str, float] | None = None
    meta: dict = {}


def _canonical_json(record: Record) -> str:
    """Return a stable, sorted-key JSON string for a Record."""
    return json.dumps(record.model_dump(), sort_keys=True, default=str)


def content_hash(record: Record) -> str:
    """SHA-256 of the canonical JSON representation of a Record."""
    return hashlib.sha256(_canonical_json(record).encode()).hexdigest()


class Dataset:
    """An ordered collection of Records with I/O and split utilities."""

    def __init__(self, records: list[Record]) -> None:
        self.records = list(records)

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, idx: int | slice) -> Record | list[Record]:
        return self.records[idx]

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_examples(self) -> list[Example]:
        """Convert each Record to a core Example."""
        examples: list[Example] = []
        for record in self.records:
            transcript = "\n".join(f"{t.role}: {t.content}" for t in record.conversation)
            inputs: dict[str, str] = {"conversation": transcript}
            labels: dict[str, str] = {}
            if record.reference_output is not None:
                labels = {"reference": record.reference_output}
            meta: dict[str, Any] = dict(record.meta)
            if record.human_label is not None:
                meta["human_label"] = record.human_label
            examples.append(Example(inputs=inputs, labels=labels, meta=meta))
        return examples

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str | pathlib.Path) -> Dataset:
        """Read one Record per line from a JSONL file."""
        path = pathlib.Path(path)
        records: list[Record] = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(Record.model_validate_json(line))
        return cls(records)

    def to_jsonl(self, path: str | pathlib.Path) -> None:
        """Write one Record per line to a JSONL file."""
        path = pathlib.Path(path)
        with path.open("w") as fh:
            for record in self.records:
                fh.write(record.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(self, fractions: dict[str, float], seed: int = 0) -> dict[str, Dataset]:
        """Deterministically partition by content-hash.

        Each record is assigned based on sha256(seed + canonical JSON) mapped
        to [0, 1).  Assignment is stable across runs and independent of record
        order.

        Parameters
        ----------
        fractions:
            Mapping of split name → fraction.  Must sum to 1.0 ± 1e-6.
        seed:
            Integer mixed into the hash to allow reproducible shuffles.

        Returns
        -------
        dict[str, Dataset]
        """
        total = sum(fractions.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Fractions must sum to 1.0 (got {total:.10f}).  "
                "Adjust your fractions so they sum to exactly 1.0."
            )

        keys = list(fractions.keys())
        # Build cumulative upper boundaries
        boundaries: list[tuple[str, float]] = []
        cumulative = 0.0
        for key in keys:
            cumulative += fractions[key]
            boundaries.append((key, cumulative))

        splits: dict[str, list[Record]] = {k: [] for k in keys}
        max_hash = 2**256

        for record in self.records:
            canonical = _canonical_json(record)
            digest = hashlib.sha256(f"{seed}{canonical}".encode()).hexdigest()
            value = int(digest, 16) / max_hash

            assigned = keys[-1]  # fallback for floating-point edge
            for key, boundary in boundaries:
                if value < boundary:
                    assigned = key
                    break
            splits[assigned].append(record)

        return {k: Dataset(v) for k, v in splits.items()}


# ------------------------------------------------------------------
# Contamination utilities
# ------------------------------------------------------------------


def contamination_check(a: Dataset, b: Dataset) -> list[str]:
    """Return content hashes that appear in both datasets."""
    hashes_a = {content_hash(r) for r in a}
    hashes_b = {content_hash(r) for r in b}
    return sorted(hashes_a & hashes_b)
