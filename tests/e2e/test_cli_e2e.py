"""Full CLI chain, end to end, in a tmp workspace via CliRunner.

init -> calibrate -> optimize (gepa) -> registry activate baseline ->
gate --candidate -> serve (app factory + TestClient).

Everything runs against PROMPTLINE_FAKE_SCRIPT keyed scripts.  The config
lowers ``gate.min_examples`` to 10 so the small dev (20) / val (15) fixtures
clear the gate's size refusals — mirroring the documented toy-run advice in
examples/support-assistant/README.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from promptline.cli.main import app, build_app_from_config
from promptline.core.types import Candidate, ModuleState
from promptline.data.dataset import Dataset
from promptline.judge.calibrator import CalibrationCertificate
from promptline.registry.registry import PromptRegistry

runner = CliRunner()

PROGRAM = "support"
SEED_INSTRUCTION = "You are a support agent. Answer the question."
#: Sentinel the scripted reflection writes into the improved instruction; the
#: keyed fake script answers RIGHT whenever it appears in the system prompt.
IMPROVED_SENTINEL = "MARKER-XYZ"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _write_config(path: Path, registry_dir: Path) -> None:
    cfg = {
        "program": {
            "name": PROGRAM,
            "instruction": SEED_INSTRUCTION,
            "inputs": ["question"],
            "outputs": ["answer"],
        },
        "models": {"task": "fake/model", "reflection": "", "judge": ""},
        # The e2e chain scripts exact-match answers; the judge metric path is
        # covered by dedicated CLI/judge tests.
        "judge": {"enabled": False},
        "dataset": {"kind": "jsonl", "path": "train.jsonl"},
        "budget": {"max_rollouts": 30, "max_cost_usd": None},
        # min_examples lowered to fit the small offline fixtures (see module
        # docstring); alpha stays at the production default.
        "gate": {"alpha": 0.05, "min_examples": 10},
        "registry": {"path": str(registry_dir)},
    }
    path.write_text(yaml.dump(cfg))


def _gold_rows(n: int = 30) -> list[dict]:
    return [
        {
            "conversation": [{"role": "user", "content": f"gold question {i}"}],
            "reference_output": f"gold answer {i}",
            "human_label": float(i % 5 + 1),
        }
        for i in range(n)
    ]


def test_cli_full_chain(tmp_path: Path) -> None:
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        _run_chain(tmp_path)
    finally:
        os.chdir(original_cwd)


def _run_chain(tmp_path: Path) -> None:
    registry_dir = tmp_path / "reg"

    # ---- init: starter config is written, then replaced by the test config --
    result = runner.invoke(app, ["init"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert (tmp_path / "promptline.yaml").exists()
    _write_config(tmp_path / "promptline.yaml", registry_dir)

    # ---- datasets -----------------------------------------------------------
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(gold_path, _gold_rows(30))
    _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"inputs": {"question": f"train question {i}"}, "labels": {"answer": "RIGHT"}}
            for i in range(8)
        ],
    )
    _write_jsonl(
        tmp_path / "dev.jsonl",
        [
            {"inputs": {"question": f"dev question {i}"}, "labels": {"answer": "RIGHT"}}
            for i in range(20)
        ],
    )
    _write_jsonl(
        tmp_path / "val.jsonl",
        [
            {"inputs": {"question": f"val question {i}"}, "labels": {"answer": "RIGHT"}}
            for i in range(15)
        ],
    )

    # ---- calibrate: judge echoes each holdout human label (kappa == 1) ------
    holdout = Dataset.from_jsonl(gold_path).split({"dev": 0.5, "holdout": 0.5}, seed=0)["holdout"]
    calibrate_script = tmp_path / "fake_calibrate.json"
    calibrate_script.write_text(
        json.dumps(
            {"responses": [f"[[reasoning]]: ok\n[[score]]: {int(r.human_label)}" for r in holdout]}
        )
    )
    result = runner.invoke(
        app,
        ["calibrate", "--gold", str(gold_path), "--criterion", "helpfulness"],
        env={**os.environ, "PROMPTLINE_FAKE_SCRIPT": str(calibrate_script)},
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    cert = CalibrationCertificate.load(registry_dir / "certificates" / "helpfulness.json")
    assert cert.passed and cert.kappa >= 0.6

    # ---- optimize (gepa): scripted reflection injects the sentinel ----------
    task_script = tmp_path / "fake_task.json"
    task_script.write_text(
        json.dumps(
            {
                # Order matters: reflection calls also quote instructions, so
                # the reflection rule must win before the sentinel task rule.
                "keyed": [
                    {
                        "contains": "Diagnose the failures",
                        "response": (
                            "The answers are wrong.\n"
                            f"```\nAlways answer RIGHT. {IMPROVED_SENTINEL}\n```"
                        ),
                    },
                    {"contains": IMPROVED_SENTINEL, "response": "[[answer]]: RIGHT"},
                ],
                "responses": ["[[answer]]: WRONG"],
            }
        )
    )
    task_env = {**os.environ, "PROMPTLINE_FAKE_SCRIPT": str(task_script)}
    result = runner.invoke(
        app,
        ["optimize", "--optimizer", "gepa", "--data", "train.jsonl", "--budget", "30"],
        env=task_env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "Registered prompt:" in result.output

    registry = PromptRegistry(registry_dir)
    prompts = registry.list_prompts(PROGRAM)
    assert len(prompts) == 1
    best_id = prompts[0]["id"]
    best = registry.get(best_id)
    assert best is not None
    assert IMPROVED_SENTINEL in best.modules[PROGRAM].instruction

    # ---- baseline: register + activate the seed prompt ----------------------
    baseline = Candidate(
        id="baseline-1",
        modules={PROGRAM: ModuleState(instruction=SEED_INSTRUCTION)},
    )
    registry.register(baseline, PROGRAM)
    result = runner.invoke(
        app,
        ["registry", "activate", "baseline-1"],
        env=task_env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    active = registry.get_active(PROGRAM)
    assert active is not None and active[0] == "baseline-1"

    # ---- gate: optimized candidate beats the baseline -> promoted -----------
    result = runner.invoke(
        app,
        ["gate", "--candidate", best_id, "--dev", "dev.jsonl", "--val", "val.jsonl"],
        env=task_env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "promote" in result.output.lower()

    active = registry.get_active(PROGRAM)
    assert active is not None and active[0] == best_id
    reports = list((registry_dir / "gate_reports").glob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["winner_id"] == best_id

    # ---- serve: the app factory serves the promoted prompt ------------------
    os.environ["PROMPTLINE_FAKE_SCRIPT"] = str(task_script)
    try:
        server_app = build_app_from_config("promptline.yaml")
    finally:
        os.environ.pop("PROMPTLINE_FAKE_SCRIPT", None)
    with TestClient(server_app) as http:
        resp = http.get(f"/prompts/{PROGRAM}/active")
        assert resp.status_code == 200
        assert resp.json()["prompt_id"] == best_id
        etag = resp.headers["ETag"]
        assert (
            http.get(f"/prompts/{PROGRAM}/active", headers={"If-None-Match": etag}).status_code
            == 304
        )
        # The calibration certificate is visible on the control plane too.
        certs = http.get("/judges/certificates").json()
        assert len(certs) == 1 and certs[0]["passed"] is True
