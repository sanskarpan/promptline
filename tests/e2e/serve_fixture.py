"""Serve a seeded Promptline workspace for the dashboard Playwright suite.

Run with ``uv run python -m tests.e2e.serve_fixture`` (repo root).  Builds a
temporary workspace containing:

* a finished GEPA run (``fixture-run``) with a realistic ``events.jsonl``
  produced by actually running the optimizer against a scripted FakeLLMClient,
* a registry with two prompts (seed + optimized winner), the winner ACTIVE
  with a real gate report attached,
* one passing judge calibration certificate,

then serves the API + built dashboard (``web/dist``) on port
``PROMPTLINE_FIXTURE_PORT`` (default 8788).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import uvicorn

from promptline.eval.harness import Budget
from promptline.judge.calibrator import Calibrator
from promptline.judge.judge import PointwiseJudge, RubricCriterion
from promptline.optimizers.gepa import GEPA
from promptline.registry.gate import GateSettings, run_gate
from promptline.registry.registry import PromptRegistry
from promptline.server.app import create_app
from promptline.server.runs import RunInfo, RunManager
from tests.e2e.conftest import (
    build_gold_dataset,
    make_harness,
    make_judge_client,
    make_pipeline_client,
    marker_metric,
    seed_for,
    support_program,
    support_trainset,
)

RUN_ID = "fixture-run"
PROGRAM = "support"


async def _seed_workspace(workspace: Path) -> tuple[PromptRegistry, RunManager]:
    program = support_program()
    seed = seed_for(program)
    client = make_pipeline_client()

    # ---- Finished optimizer run with realistic events.jsonl ----------------
    runs_dir = workspace / "runs"
    run_dir = runs_dir / RUN_ID
    result = await GEPA(
        minibatch_size=3, max_iterations=4, use_merge=False, run_dir=run_dir
    ).optimize(
        program=program,
        seed=seed,
        trainset=support_trainset(20, "fixture"),
        metric=marker_metric,
        budget=Budget(max_rollouts=200),
        harness=make_harness(client),
    )
    best = result.best

    # ---- Gate report for the activation -------------------------------------
    report = await run_gate(
        program=program,
        incumbent=seed,
        candidates=[best],
        dev=support_trainset(60, "dev"),
        val=support_trainset(50, "val"),
        harness=make_harness(client),
        metric=marker_metric,
        settings=GateSettings(min_examples=50),
    )

    # ---- Registry: 2 prompts, winner active ----------------------------------
    registry = PromptRegistry(workspace / "registry")
    registry.register(seed, PROGRAM)
    registry.register(best, PROGRAM, run_id=RUN_ID)
    registry.record_eval(seed.id, "fixture", result.scores[seed.id], n=20)
    registry.record_eval(best.id, "fixture", result.scores[best.id], n=20)
    registry.activate(PROGRAM, seed.id)
    registry.activate(PROGRAM, best.id, report.model_dump_json())

    # ---- Passing judge certificate -------------------------------------------
    judge = PointwiseJudge(
        criterion=RubricCriterion(
            name="helpfulness",
            description="How well the response addresses the request.",
        ),
        judge_model="fake/judge",
    )
    cert = await Calibrator(judge, build_gold_dataset(30), make_judge_client("high")).calibrate()
    cert.save(registry.root / "certificates" / "helpfulness.json")

    # ---- RunManager seeded with the finished run ------------------------------
    run_manager = RunManager(runs_dir)
    run_manager._runs[RUN_ID] = RunInfo(  # noqa: SLF001 — test fixture seeding
        run_id=RUN_ID,
        status="finished",
        summary={"best_id": best.id, "best_score": result.scores[best.id]},
    )
    return registry, run_manager


def main() -> None:
    port = int(os.environ.get("PROMPTLINE_FIXTURE_PORT", "8788"))
    workspace = Path(tempfile.mkdtemp(prefix="promptline-e2e-"))
    registry, run_manager = asyncio.run(_seed_workspace(workspace))

    web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if not (web_dist / "index.html").exists():
        raise SystemExit("web/dist/index.html not found — run `npm run build` in web/ first")

    app = create_app(registry, run_manager, web_dist=web_dist)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
