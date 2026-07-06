"""Tests for the `promptline demo` sub-app (offline mode only — hermetic)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from promptline.cli.main import app
from promptline.core.config import PromptlineConfig, load_config
from promptline.data.dataset import Dataset, contamination_check

runner = CliRunner()

SPLIT_FILES = ["gold.jsonl", "dev.jsonl", "val.jsonl", "feedback.jsonl", "train.jsonl"]


def _setup_offline(tmp_path: Path, extra: list[str] | None = None) -> Path:
    workspace = tmp_path / "ws"
    result = runner.invoke(
        app,
        ["demo", "setup", "--dir", str(workspace), "--offline", *(extra or [])],
    )
    assert result.exit_code == 0, result.output
    return workspace


class TestDemoSetupOffline:
    def test_writes_all_files(self, tmp_path: Path) -> None:
        workspace = _setup_offline(tmp_path)
        for name in SPLIT_FILES:
            assert (workspace / name).exists(), f"missing {name}"
        assert (workspace / "promptline.yaml").exists()

    def test_config_parses_and_has_demo_shape(self, tmp_path: Path) -> None:
        workspace = _setup_offline(tmp_path)
        cfg = load_config(workspace / "promptline.yaml")
        assert isinstance(cfg, PromptlineConfig)
        assert cfg.program.instruction == ("You are a support agent. Answer the question.")
        assert cfg.models.task == "meta-llama/llama-3.1-8b-instruct"
        assert cfg.models.reflection == "anthropic/claude-3.5-haiku"
        assert cfg.models.judge == "anthropic/claude-3.5-haiku"
        assert cfg.budget.max_rollouts == 300
        assert cfg.budget.max_cost_usd == 5.0
        assert cfg.gate.min_examples == 50
        assert cfg.dataset.path == "train.jsonl"

    def test_splits_are_disjoint(self, tmp_path: Path) -> None:
        workspace = _setup_offline(tmp_path)
        dev = Dataset.from_jsonl(workspace / "dev.jsonl")
        val = Dataset.from_jsonl(workspace / "val.jsonl")
        assert len(dev) > 0
        assert len(val) > 0
        assert contamination_check(dev, val) == []

    def test_gold_has_numeric_human_labels(self, tmp_path: Path) -> None:
        workspace = _setup_offline(tmp_path)
        gold = Dataset.from_jsonl(workspace / "gold.jsonl")
        assert len(gold) > 0
        for record in gold:
            assert isinstance(record.human_label, float)
            assert 0.0 <= record.human_label <= 4.0
            assert record.reference_output

    def test_gold_n_limits_gold_set(self, tmp_path: Path) -> None:
        workspace = _setup_offline(tmp_path, ["--gold-n", "10"])
        gold = Dataset.from_jsonl(workspace / "gold.jsonl")
        assert len(gold) == 10

    def test_train_is_optimize_format(self, tmp_path: Path) -> None:
        workspace = _setup_offline(tmp_path)
        lines = (workspace / "train.jsonl").read_text().splitlines()
        assert lines
        for line in lines:
            row = json.loads(line)
            assert "conversation" in row["inputs"]
            assert "reference" in row["labels"]

    def test_dev_val_also_carry_optimize_fields(self, tmp_path: Path) -> None:
        """dev/val are Record-format but embed inputs/labels for `gate`."""
        workspace = _setup_offline(tmp_path)
        for name in ("dev.jsonl", "val.jsonl"):
            for line in (workspace / name).read_text().splitlines():
                row = json.loads(line)
                assert "conversation" in row["inputs"]
                assert "reference" in row["labels"]


class TestDataPrepareAlias:
    def test_data_prepare_demo_forwards(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        result = runner.invoke(
            app,
            ["data", "prepare", "--demo", "--offline", "--dir", str(workspace)],
        )
        assert result.exit_code == 0, result.output
        for name in SPLIT_FILES:
            assert (workspace / name).exists()

    def test_data_prepare_without_demo_is_noop(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["data", "prepare"])
        assert result.exit_code == 0
