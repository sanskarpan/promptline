"""Row-mappers and public dataset loaders for HuggingFace datasets.

The pure ``_map_*`` functions have no dependency on the ``datasets`` package and
are unit-testable in isolation.  The public ``load_*`` functions guard their
``import datasets`` call so that the rest of the package remains usable when the
optional ``promptline[data]`` extra is not installed.
"""

from __future__ import annotations

from typing import Any

from promptline.data.dataset import Dataset, Record, Turn

# ---------------------------------------------------------------------------
# Pure row-mappers (no `datasets` dependency)
# ---------------------------------------------------------------------------


def _map_helpsteer_row(row: dict[str, Any], attribute: str = "helpfulness") -> Record:
    """Map a single HelpSteer2 row to a Record.

    HelpSteer2 schema::

        {
            "prompt": str,
            "response": str,
            "helpfulness": int (0-4),
            "correctness": int,
            "coherence": int,
            "complexity": int,
            "verbosity": int,
        }
    """
    _ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
    all_ratings = {attr: row[attr] for attr in _ATTRIBUTES}
    return Record(
        conversation=[Turn(role="user", content=row["prompt"])],
        reference_output=row["response"],
        human_label=float(row[attribute]),
        meta={"attribute": attribute, "all_ratings": all_ratings},
    )


def _map_bitext_row(row: dict[str, Any]) -> Record:
    """Map a single Bitext customer-support row to a Record.

    Bitext schema::

        {
            "instruction": str,
            "response": str,
            "category": str,
            "intent": str,
            "flags": str,
        }
    """
    return Record(
        conversation=[Turn(role="user", content=row["instruction"])],
        reference_output=row["response"],
        meta={
            "category": row["category"],
            "intent": row["intent"],
            "flags": row.get("flags", ""),
        },
    )


def _map_mtbench_row(row: dict[str, Any]) -> Record:
    """Map a single lmsys/mt_bench_human_judgments row to a Record.

    MT-Bench schema::

        {
            "question_id": int,
            "model_a": str,
            "model_b": str,
            "winner": "model_a" | "model_b" | "tie" | "tie (bothbad)",
            "judge": str,
            "conversation_a": [{"role": str, "content": str}, ...],
            "conversation_b": [{"role": str, "content": str}, ...],
        }

    Only the *user* turns from ``conversation_a`` are included in the
    conversation.  The last assistant turn from each conversation is stored in
    ``meta`` for later use by an LLM judge.
    """

    def _last_assistant(turns: list[dict[str, str]]) -> str:
        for turn in reversed(turns):
            if turn.get("role") == "assistant":
                return turn["content"]
        return ""

    conv_a: list[dict[str, str]] = row["conversation_a"]
    conv_b: list[dict[str, str]] = row["conversation_b"]

    user_turns = [
        Turn(role=t["role"], content=t["content"]) for t in conv_a if t["role"] == "user"
    ]

    return Record(
        conversation=user_turns,
        reference_output=None,
        human_label=None,
        meta={
            "response_a": _last_assistant(conv_a),
            "response_b": _last_assistant(conv_b),
            "winner": row["winner"],
            "question_id": row["question_id"],
            "judge": row["judge"],
            "turn": row.get("turn"),
        },
    )


# ---------------------------------------------------------------------------
# Public loaders (require ``datasets``)
# ---------------------------------------------------------------------------

_DATA_EXTRA_MSG = (
    "The 'datasets' package is required to use this loader.  "
    "Install it with:  pip install 'promptline[data]'"
)


def _apply_limit(ds: Any, limit: int | None) -> Any:
    """Slice a HuggingFace dataset to *limit* rows BEFORE row-mapping."""
    if limit is not None:
        return ds.select(range(min(limit, len(ds))))
    return ds


def load_helpsteer2(
    split: str = "train",
    attribute: str = "helpfulness",
    limit: int | None = None,
) -> Dataset:
    """Load nvidia/HelpSteer2 and return a Dataset of Records.

    Requires ``promptline[data]``.
    """
    try:
        import datasets as hf_datasets
    except ImportError as exc:
        raise ImportError(_DATA_EXTRA_MSG) from exc

    ds = _apply_limit(hf_datasets.load_dataset("nvidia/HelpSteer2", split=split), limit)
    records = [_map_helpsteer_row(dict(row), attribute=attribute) for row in ds]
    return Dataset(records)


def load_bitext(limit: int | None = None) -> Dataset:
    """Load bitext/Bitext-customer-support-llm-chatbot-training-dataset.

    Requires ``promptline[data]``.
    """
    try:
        import datasets as hf_datasets
    except ImportError as exc:
        raise ImportError(_DATA_EXTRA_MSG) from exc

    ds = _apply_limit(
        hf_datasets.load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            split="train",
        ),
        limit,
    )
    records = [_map_bitext_row(dict(row)) for row in ds]
    return Dataset(records)


def load_mtbench_human(limit: int | None = None) -> Dataset:
    """Load lmsys/mt_bench_human_judgments.

    Requires ``promptline[data]``.
    """
    try:
        import datasets as hf_datasets
    except ImportError as exc:
        raise ImportError(_DATA_EXTRA_MSG) from exc

    ds = _apply_limit(
        hf_datasets.load_dataset("lmsys/mt_bench_human_judgments", split="human"),
        limit,
    )
    records = [_map_mtbench_row(dict(row)) for row in ds]
    return Dataset(records)
