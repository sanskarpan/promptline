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


def test_optimize_bootstrap_rs_budget_zero_exits_cleanly(tmp_path: Path) -> None:
    """optimize --optimizer bootstrap-rs --budget 0 must exit 0, no traceback.

    Finding 1 (CLI crash): result.scores may be empty when budget=0 causes the
    subset-evaluation loop to be skipped entirely.  The CLI must not attempt
    ``f"...{value:.3f}"`` when *value* is the string ``'?'`` (the old fallback).
    """
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
            "--optimizer", "bootstrap-rs",
            "--config", str(cfg_path),
            "--data", str(data_path),
            "--budget", "0",
        ],
        env=env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, (
        f"Expected exit 0 with --budget 0. Got:\n{result.output}"
    )
    assert "Traceback" not in result.output


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


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def _write_gold_jsonl(path: Path, n: int = 30) -> None:
    """Gold records with human labels cycling 1..5."""
    rows = []
    for i in range(n):
        label = i % 5 + 1
        rows.append(
            {
                "conversation": [{"role": "user", "content": f"question {i}"}],
                "reference_output": f"answer {i}",
                "human_label": float(label),
            }
        )
    _write_jsonl(path, rows)


def _holdout_records(gold_path: Path):
    """Replicate the Calibrator's default split to learn the holdout order."""
    from promptline.data.dataset import Dataset

    dataset = Dataset.from_jsonl(gold_path)
    return dataset.split({"dev": 0.5, "holdout": 0.5}, seed=0)["holdout"]


def test_calibrate_perfect_agreement_exit_0_and_saves_cert(tmp_path: Path) -> None:
    cfg_path = tmp_path / "promptline.yaml"
    gold_path = tmp_path / "gold.jsonl"
    fake_path = tmp_path / "fake_script.json"

    _write_config(cfg_path)
    _write_gold_jsonl(gold_path)

    holdout = _holdout_records(gold_path)
    labels = sorted({int(r.human_label) for r in holdout})
    assert labels == [1, 2, 3, 4, 5], "holdout must span the judge scale"
    # Judge echoes each holdout human label, in holdout order.
    _write_fake_script(
        fake_path,
        [
            f"[[reasoning]]: ok\n[[score]]: {int(r.human_label)}"
            for r in holdout
        ],
    )

    env = {**os.environ, "PROMPTLINE_FAKE_SCRIPT": str(fake_path)}
    with _in_tmpdir(tmp_path):
        result = runner.invoke(
            app,
            [
                "calibrate",
                "--gold", str(gold_path),
                "--criterion", "helpfulness",
                "--threshold", "0.6",
                "--config", str(cfg_path),
            ],
            env=env,
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output

    cert_path = tmp_path / ".promptline_test" / "certificates" / "helpfulness.json"
    assert cert_path.exists()
    from promptline.judge.calibrator import CalibrationCertificate

    cert = CalibrationCertificate.load(cert_path)
    assert cert.passed is True
    assert cert.kappa == 1.0


def test_calibrate_disagreement_exit_1(tmp_path: Path) -> None:
    cfg_path = tmp_path / "promptline.yaml"
    gold_path = tmp_path / "gold.jsonl"
    fake_path = tmp_path / "fake_script.json"

    _write_config(cfg_path)
    _write_gold_jsonl(gold_path)
    # Judge always answers 1 while human labels vary -> kappa ~ 0.
    _write_fake_script(fake_path, ["[[reasoning]]: r\n[[score]]: 1"])

    env = {**os.environ, "PROMPTLINE_FAKE_SCRIPT": str(fake_path)}
    with _in_tmpdir(tmp_path):
        result = runner.invoke(
            app,
            [
                "calibrate",
                "--gold", str(gold_path),
                "--criterion", "helpfulness",
                "--config", str(cfg_path),
            ],
            env=env,
        )
    assert result.exit_code == 1, result.output
    # Certificate is still saved for inspection, marked failed.
    cert_path = tmp_path / ".promptline_test" / "certificates" / "helpfulness.json"
    assert cert_path.exists()


def test_calibrate_respects_n_limit(tmp_path: Path) -> None:
    cfg_path = tmp_path / "promptline.yaml"
    gold_path = tmp_path / "gold.jsonl"
    fake_path = tmp_path / "fake_script.json"

    _write_config(cfg_path)
    _write_gold_jsonl(gold_path, n=30)

    from promptline.data.dataset import Dataset

    limited = Dataset(Dataset.from_jsonl(gold_path).records[:10])
    holdout = limited.split({"dev": 0.5, "holdout": 0.5}, seed=0)["holdout"]
    _write_fake_script(
        fake_path,
        [
            f"[[reasoning]]: ok\n[[score]]: {int(r.human_label)}"
            for r in holdout
        ],
    )

    env = {**os.environ, "PROMPTLINE_FAKE_SCRIPT": str(fake_path)}
    with _in_tmpdir(tmp_path):
        result = runner.invoke(
            app,
            [
                "calibrate",
                "--gold", str(gold_path),
                "--n", "10",
                "--config", str(cfg_path),
            ],
            env=env,
            catch_exceptions=False,
        )
    # n_holdout in the saved certificate must reflect the limited dataset.
    from promptline.judge.calibrator import CalibrationCertificate

    cert_path = tmp_path / ".promptline_test" / "certificates" / "helpfulness.json"
    assert cert_path.exists(), result.output
    cert = CalibrationCertificate.load(cert_path)
    assert cert.n_holdout == len(holdout)


def test_calibrate_missing_gold_exits_nonzero(tmp_path: Path) -> None:
    cfg_path = tmp_path / "promptline.yaml"
    _write_config(cfg_path)
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--gold", str(tmp_path / "missing.jsonl"),
            "--config", str(cfg_path),
        ],
    )
    assert result.exit_code != 0
