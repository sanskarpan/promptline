"""CLI tests using typer.testing.CliRunner."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import yaml
from typer.testing import CliRunner

from promptline.cli.main import app
from promptline.core.config import load_config

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _in_tmpdir(tmp_path: Path):
    """Context manager: chdir into tmp_path, restore on exit."""
    original = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(original)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_fake_script(path: Path, responses: list[str]) -> None:
    path.write_text(json.dumps({"responses": responses}))


def _write_config(path: Path, instruction: str = "Answer the question.") -> None:
    cfg = {
        "program": {
            "name": "main",
            "instruction": instruction,
            "inputs": ["question"],
            "outputs": ["answer"],
        },
        "models": {"task": "fake/model", "reflection": "", "judge": ""},
        "dataset": {"kind": "jsonl", "path": ""},
        "budget": {"max_rollouts": 20, "max_cost_usd": None},
        "registry": {"path": ".promptline_test"},
    }
    path.write_text(yaml.dump(cfg))


# ---------------------------------------------------------------------------
# init tests
# ---------------------------------------------------------------------------


def test_init_creates_config(tmp_path: Path) -> None:
    """init should write a parseable promptline.yaml."""
    with _in_tmpdir(tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.output
        config_path = tmp_path / "promptline.yaml"
        assert config_path.exists()
        cfg = load_config(config_path)
        assert cfg.program.inputs == ["question"]
        assert cfg.program.outputs == ["answer"]


def test_init_refuses_overwrite(tmp_path: Path) -> None:
    """init without --force should refuse to overwrite an existing config."""
    with _in_tmpdir(tmp_path):
        runner.invoke(app, ["init"])  # first time
        result = runner.invoke(app, ["init"])  # second time
        assert result.exit_code != 0
        assert "already exists" in result.output


def test_init_force_overwrites(tmp_path: Path) -> None:
    """init --force should overwrite an existing config."""
    with _in_tmpdir(tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["init", "--force"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# version test
# ---------------------------------------------------------------------------


def test_version_prints_version() -> None:
    """version command should print the package version string."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    from promptline import __version__
    assert __version__ in result.output


# ---------------------------------------------------------------------------
# optimize tests
# ---------------------------------------------------------------------------


def _make_examples(n: int = 3) -> list[dict]:
    """Create n examples where inputs.question and labels.answer are set."""
    return [
        {"inputs": {"question": f"q{i}"}, "labels": {"answer": f"q{i}"}}
        for i in range(n)
    ]


def _make_fake_responses(n_examples: int = 3) -> list[str]:
    """Cycle of responses that parse correctly and match labels."""
    return [f"[[answer]]: q{i}" for i in range(n_examples)] * 4


def test_optimize_bootstrap_exit_0(tmp_path: Path) -> None:
    """optimize with bootstrap optimizer should exit 0 and print a score."""
    cfg_path = tmp_path / "promptline.yaml"
    data_path = tmp_path / "data.jsonl"
    fake_path = tmp_path / "fake_script.json"

    _write_config(cfg_path)
    _write_jsonl(data_path, _make_examples(3))
    _write_fake_script(fake_path, _make_fake_responses(3))

    env = {**os.environ, "PROMPTLINE_FAKE_SCRIPT": str(fake_path)}
    result = runner.invoke(
        app,
        [
            "optimize",
            "--optimizer", "bootstrap",
            "--config", str(cfg_path),
            "--data", str(data_path),
            "--budget", "10",
        ],
        env=env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Output should contain a score (some decimal number).
    import re
    assert re.search(r"\d+\.\d+", result.output), (
        f"Expected a numeric score in output. Got:\n{result.output}"
    )


def test_optimize_opro_exit_0(tmp_path: Path) -> None:
    """optimize with OPRO optimizer should exit 0."""
    cfg_path = tmp_path / "promptline.yaml"
    data_path = tmp_path / "data.jsonl"
    fake_path = tmp_path / "fake_script.json"

    _write_config(cfg_path)
    _write_jsonl(data_path, _make_examples(3))

    # OPRO needs task responses for seed eval + proposer responses for steps.
    responses: list[str] = []
    # Seed eval: 3 examples.
    for i in range(3):
        responses.append(f"[[answer]]: q{i}")
    # Proposer responses: <INS> blocks + eval responses, interleaved cycling.
    for _ in range(30):
        responses.append("<INS>Improved instruction.</INS>")
    for i in range(3):
        responses.append(f"[[answer]]: q{i}")

    _write_fake_script(fake_path, responses)

    env = {**os.environ, "PROMPTLINE_FAKE_SCRIPT": str(fake_path)}
    result = runner.invoke(
        app,
        [
            "optimize",
            "--optimizer", "opro",
            "--config", str(cfg_path),
            "--data", str(data_path),
            "--budget", "10",
        ],
        env=env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output


def test_optimize_missing_config(tmp_path: Path) -> None:
    """optimize should exit non-zero when config file is missing."""
    result = runner.invoke(
        app,
        [
            "optimize",
            "--config", str(tmp_path / "nonexistent.yaml"),
            "--data", str(tmp_path / "d.jsonl"),
        ],
    )
    assert result.exit_code != 0


def test_optimize_missing_data(tmp_path: Path) -> None:
    """optimize should exit non-zero when data file is missing."""
    cfg_path = tmp_path / "promptline.yaml"
    _write_config(cfg_path)
    result = runner.invoke(
        app,
        [
            "optimize",
            "--config", str(cfg_path),
            "--data", str(tmp_path / "missing.jsonl"),
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# load_examples_jsonl
# ---------------------------------------------------------------------------


def test_load_examples_jsonl(tmp_path: Path) -> None:
    """load_examples_jsonl should parse JSONL into Example objects."""
    from promptline.cli.main import load_examples_jsonl

    data_path = tmp_path / "data.jsonl"
    _write_jsonl(
        data_path,
        [
            {"inputs": {"question": "q1"}, "labels": {"answer": "a1"}},
            {"inputs": {"question": "q2"}, "labels": {"answer": "a2"}},
        ],
    )
    examples = load_examples_jsonl(str(data_path))
    assert len(examples) == 2
    assert examples[0].inputs == {"question": "q1"}
    assert examples[0].labels == {"answer": "a1"}
    assert examples[1].inputs == {"question": "q2"}
