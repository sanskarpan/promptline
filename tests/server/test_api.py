"""FastAPI server tests (TestClient against a tmp registry)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from promptline.core.types import Candidate, ModuleState
from promptline.optimizers.base import RunEvent
from promptline.registry.registry import PromptRegistry
from promptline.server.app import create_app
from promptline.server.runs import RunManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cand(cand_id: str, instruction: str = "Answer.") -> Candidate:
    return Candidate(
        id=cand_id, modules={"main": ModuleState(instruction=instruction)}
    )


def _make_app(tmp_path: Path, run_starter=None, gate_runner=None):
    registry = PromptRegistry(tmp_path / "registry")
    run_manager = RunManager(tmp_path / "runs")
    app = create_app(
        registry, run_manager, run_starter=run_starter, gate_runner=gate_runner
    )
    return app, registry, run_manager


def _fake_run_starter(spec, emit, run_dir=None):
    """Fake run: emit 3 events then finish with a best-candidate result."""

    async def _run():
        for i in range(3):
            emit(RunEvent.now("budget_tick", step=i))

        class _Result:
            best = _cand("best-1")
            scores = {"best-1": 0.9}

        return _Result()

    return _run()


def _wait_for_status(client: TestClient, run_id: str, status: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = client.get(f"/runs/{run_id}").json()
        if detail["status"] == status:
            return detail
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never reached status {status!r}")


# ---------------------------------------------------------------------------
# Serving plane
# ---------------------------------------------------------------------------


def test_active_prompt_404_then_200_etag_then_304(tmp_path: Path) -> None:
    app, registry, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/prompts/main/active").status_code == 404

        cand = _cand("p1", instruction="Be helpful.")
        registry.register(cand, "main")
        registry.activate("main", "p1")

        resp = client.get("/prompts/main/active")
        assert resp.status_code == 200
        # RFC 7232: ETag values must be quoted strings.
        assert resp.headers["etag"] == '"p1"'
        body = resp.json()
        assert body["program"] == "main"
        assert body["prompt_id"] == "p1"
        assert body["modules"]["main"]["instruction"] == "Be helpful."
        assert body["activated_at"]

        # If-None-Match with quoted ETag → 304.
        cached = client.get(
            "/prompts/main/active", headers={"If-None-Match": '"p1"'}
        )
        assert cached.status_code == 304
        assert not cached.content

        # Bare (unquoted) If-None-Match also matches (strip on compare).
        cached_bare = client.get(
            "/prompts/main/active", headers={"If-None-Match": "p1"}
        )
        assert cached_bare.status_code == 304

        stale = client.get(
            "/prompts/main/active", headers={"If-None-Match": '"other"'}
        )
        assert stale.status_code == 200


# ---------------------------------------------------------------------------
# Runs lifecycle + SSE
# ---------------------------------------------------------------------------


def test_runs_lifecycle_and_sse_replay(tmp_path: Path) -> None:
    app, _, run_manager = _make_app(tmp_path, run_starter=_fake_run_starter)
    with TestClient(app) as client:
        resp = client.post("/runs", json={"optimizer": "bootstrap"})
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        detail = _wait_for_status(client, run_id, "finished")
        assert detail["summary"] == {"best_id": "best-1", "best_score": 0.9}

        listing = client.get("/runs").json()
        assert [r["run_id"] for r in listing] == [run_id]
        assert listing[0]["status"] == "finished"

        # SSE replay: run is finished so the stream replays 3 events, closes.
        events: list[dict] = []
        with client.stream("GET", f"/runs/{run_id}/events") as stream:
            for line in stream.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[len("data:"):].strip()))
        assert len(events) == 3
        assert [e["type"] for e in events] == ["budget_tick"] * 3
        assert [e["payload"]["step"] for e in events] == [0, 1, 2]


def test_run_failure_reported(tmp_path: Path) -> None:
    def _boom_starter(spec, emit, run_dir=None):
        async def _run():
            raise RuntimeError("kaput")

        return _run()

    app, _, _ = _make_app(tmp_path, run_starter=_boom_starter)
    with TestClient(app) as client:
        run_id = client.post("/runs", json={"optimizer": "opro"}).json()["run_id"]
        detail = _wait_for_status(client, run_id, "failed")
        assert "kaput" in detail["error"]


def test_runs_unknown_and_unconfigured(tmp_path: Path) -> None:
    app, _, _ = _make_app(tmp_path)  # no run_starter
    with TestClient(app) as client:
        assert client.post("/runs", json={"optimizer": "x"}).status_code == 400
        assert client.get("/runs/nope").status_code == 404
        assert client.get("/runs/nope/events").status_code == 404


# ---------------------------------------------------------------------------
# Gate endpoint
# ---------------------------------------------------------------------------


def test_gate_endpoint_delegates_to_runner(tmp_path: Path) -> None:
    seen: list[dict] = []

    def _fake_gate(payload: dict) -> dict:
        seen.append(payload)
        return {
            "program": payload["program"],
            "incumbent_id": payload["incumbent_id"],
            "winner_id": payload["candidate_ids"][0],
            "verdict": "promote",
        }

    app, _, _ = _make_app(tmp_path, gate_runner=_fake_gate)
    with TestClient(app) as client:
        resp = client.post(
            "/gate",
            json={
                "program": "main",
                "incumbent_id": "p0",
                "candidate_ids": ["p1"],
                "dev_path": "dev.jsonl",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "promote"
        assert seen[0]["dev_path"] == "dev.jsonl"  # extra keys pass through


def test_gate_endpoint_unconfigured_400(tmp_path: Path) -> None:
    app, _, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.post(
            "/gate", json={"program": "main", "candidate_ids": []}
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Registry endpoints
# ---------------------------------------------------------------------------


def test_registry_list_activate_rollback(tmp_path: Path) -> None:
    app, registry, _ = _make_app(tmp_path)
    registry.register(_cand("p1"), "main")
    registry.register(_cand("p2"), "main")

    with TestClient(app) as client:
        listing = client.get("/registry/main").json()
        assert [p["id"] for p in listing] == ["p1", "p2"]

        # Rollback before any activation -> 409.
        assert client.post("/registry/main/rollback").status_code == 409

        # Activate unknown prompt -> 404.
        resp = client.post(
            "/registry/main/activate", json={"prompt_id": "ghost"}
        )
        assert resp.status_code == 404

        assert (
            client.post(
                "/registry/main/activate", json={"prompt_id": "p1"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/registry/main/activate",
                json={"prompt_id": "p2", "gate_report": {"verdict": "promote"}},
            ).status_code
            == 200
        )
        assert client.get("/prompts/main/active").json()["prompt_id"] == "p2"

        rolled = client.post("/registry/main/rollback")
        assert rolled.status_code == 200
        assert rolled.json()["prompt_id"] == "p1"
        assert client.get("/prompts/main/active").json()["prompt_id"] == "p1"

        # Only one distinct prior activation remains -> further rollback 409.
        assert client.post("/registry/main/rollback").status_code == 409


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------


def test_certificates_listing(tmp_path: Path) -> None:
    app, registry, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/judges/certificates").json() == []

        cert_dir = Path(registry.root) / "certificates"
        cert_dir.mkdir(parents=True)
        (cert_dir / "helpfulness.json").write_text(
            json.dumps({"criterion": "helpfulness", "kappa": 0.8, "passed": True})
        )
        certs = client.get("/judges/certificates").json()
        assert len(certs) == 1
        assert certs[0]["criterion"] == "helpfulness"


# ---------------------------------------------------------------------------
# RunManager unit behaviour
# ---------------------------------------------------------------------------


async def test_run_manager_start_get_list(tmp_path: Path) -> None:
    manager = RunManager(tmp_path)

    def factory(emit, run_dir):
        async def _run():
            emit(RunEvent.now("run_started"))
            return {"done": True}

        return _run()

    run_id = manager.start(factory, run_id="fixed-id")
    assert run_id == "fixed-id"
    assert manager.get("fixed-id")["status"] == "running"

    import asyncio

    for _ in range(100):
        if manager.get("fixed-id")["status"] != "running":
            break
        await asyncio.sleep(0.01)
    info = manager.get("fixed-id")
    assert info["status"] == "finished"
    assert info["summary"] == {"done": True}
    assert manager.events_path("fixed-id").exists()
    assert manager.get("unknown") is None
    assert [r["run_id"] for r in manager.list()] == ["fixed-id"]


# ---------------------------------------------------------------------------
# Finding 1: synchronous factory error → 400 + run shows failed
# ---------------------------------------------------------------------------


def test_post_runs_sync_factory_error_returns_400_run_shows_failed(tmp_path: Path) -> None:
    """If the coro factory raises synchronously, POST /runs → 400 (not 500).

    The run is registered with status='failed' so GET /runs/{id} works.
    The SSE stream for the failed run terminates immediately.
    """

    def _raising_starter(spec, emit, run_dir=None):
        raise FileNotFoundError(f"dataset not found: {spec.data_path}")

    app, _, _ = _make_app(tmp_path, run_starter=_raising_starter)
    with TestClient(app) as client:
        resp = client.post(
            "/runs", json={"optimizer": "bootstrap", "data_path": "/no/such.jsonl"}
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "dataset not found" in body.get("detail", "").lower()
        # The 400 body includes the run_id so the client can inspect it.
        run_id = body.get("run_id")
        assert run_id is not None

        # GET /runs/{id} shows the run as failed (not absent, not lingering running).
        run_info = client.get(f"/runs/{run_id}").json()
        assert run_info["status"] == "failed"
        assert "dataset not found" in run_info["error"].lower()

        # SSE on the failed run terminates immediately (no events, no hang).
        events: list[str] = []
        with client.stream("GET", f"/runs/{run_id}/events") as stream:
            for line in stream.iter_lines():
                if line.startswith("data:"):
                    events.append(line)
        assert events == []


# ---------------------------------------------------------------------------
# Finding 2: gate_runner errors → 400 (not 500)
# ---------------------------------------------------------------------------


def test_gate_runner_value_error_returns_400(tmp_path: Path) -> None:
    """ValueError from gate_runner (e.g. 'no incumbent') → 400."""

    def _value_error_gate(payload: dict) -> dict:
        raise ValueError("no incumbent prompt; activate a baseline first")

    app, _, _ = _make_app(tmp_path, gate_runner=_value_error_gate)
    with TestClient(app) as client:
        resp = client.post("/gate", json={"program": "main", "candidate_ids": []})
        assert resp.status_code == 400
        assert "no incumbent" in resp.json()["detail"].lower()


def test_gate_runner_key_error_returns_400(tmp_path: Path) -> None:
    """KeyError from gate_runner (e.g. accessing a missing payload key) → 400."""

    def _key_error_gate(payload: dict) -> dict:
        # Simulates a gate_runner that requires a key not declared in GateRequest.
        raise KeyError("some_required_but_missing_key")

    app, _, _ = _make_app(tmp_path, gate_runner=_key_error_gate)
    with TestClient(app) as client:
        resp = client.post("/gate", json={"program": "main", "candidate_ids": []})
        assert resp.status_code == 400


def test_gate_runner_file_not_found_returns_400(tmp_path: Path) -> None:
    """FileNotFoundError from gate_runner (bad path) → 400."""

    def _fnf_gate(payload: dict) -> dict:
        raise FileNotFoundError("no such file: /bad/path.jsonl")

    app, _, _ = _make_app(tmp_path, gate_runner=_fnf_gate)
    with TestClient(app) as client:
        resp = client.post("/gate", json={"program": "main", "candidate_ids": []})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Finding 5: server-started run registers prompt with run_id + record_eval
# ---------------------------------------------------------------------------


def test_server_run_registers_with_run_id_and_records_eval(tmp_path: Path) -> None:
    """POST /runs → finished run should register the prompt with run_id set and
    call record_eval so list_prompts shows a mean_score."""
    import json as _json
    import os

    import yaml

    from promptline.cli.main import build_app_from_config
    from promptline.registry.registry import PromptRegistry

    cfg_path = tmp_path / "promptline.yaml"
    registry_dir = tmp_path / "reg"
    cfg = {
        "program": {
            "name": "main",
            "instruction": "Answer.",
            "inputs": ["question"],
            "outputs": ["answer"],
        },
        "models": {"task": "fake/model", "reflection": "", "judge": ""},
        "dataset": {"kind": "jsonl", "path": ""},
        "judge": {"enabled": False},
        "budget": {"max_rollouts": 10, "max_cost_usd": None},
        "gate": {"alpha": 0.05, "min_examples": 50},
        "registry": {"path": str(registry_dir)},
    }
    cfg_path.write_text(yaml.dump(cfg))

    data_path = tmp_path / "data.jsonl"
    fake_path = tmp_path / "fake.json"
    rows = [
        {"inputs": {"question": f"q{i}"}, "labels": {"answer": f"q{i}"}}
        for i in range(3)
    ]
    data_path.write_text("\n".join(_json.dumps(r) for r in rows))
    fake_path.write_text(
        _json.dumps({"responses": [f"[[answer]]: q{i}" for i in range(3)] * 4})
    )

    env_backup = os.environ.copy()
    os.environ["PROMPTLINE_FAKE_SCRIPT"] = str(fake_path)
    try:
        server_app = build_app_from_config(str(cfg_path))
        with TestClient(server_app) as client:
            resp = client.post(
                "/runs",
                json={"optimizer": "bootstrap", "data_path": str(data_path)},
            )
            assert resp.status_code == 200, resp.text
            run_id = resp.json()["run_id"]
            detail = _wait_for_status(client, run_id, "finished")
            assert detail["status"] == "finished"

            reg = PromptRegistry(registry_dir)
            prompts = reg.list_prompts("main")
            assert len(prompts) >= 1
            # run_id must be recorded (not blank).
            assert prompts[0]["run_id"] == run_id
            # record_eval was called → mean_score is a float, not None.
            assert isinstance(prompts[0]["mean_score"], float)
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


# ---------------------------------------------------------------------------
# Static dashboard mount
# ---------------------------------------------------------------------------


def _make_web_dist(tmp_path: Path) -> Path:
    web_dist = tmp_path / "web_dist"
    web_dist.mkdir()
    (web_dist / "index.html").write_text("<html><body>PROMPTLINE</body></html>")
    (web_dist / "app.js").write_text("console.log('promptline')")
    return web_dist


def test_web_dist_served_at_root_and_api_still_wins(tmp_path: Path) -> None:
    registry = PromptRegistry(tmp_path / "registry")
    run_manager = RunManager(tmp_path / "runs")
    app = create_app(registry, run_manager, web_dist=_make_web_dist(tmp_path))
    with TestClient(app) as client:
        # Root serves the dashboard.
        resp = client.get("/")
        assert resp.status_code == 200
        assert "PROMPTLINE" in resp.text

        # Real static assets are served.
        assert client.get("/app.js").status_code == 200

        # SPA fallback: /ui/* paths (which never match an API route) → index.html.
        resp = client.get("/ui/runs")
        assert resp.status_code == 200
        assert "PROMPTLINE" in resp.text

        resp = client.get("/ui/lineage/some-run")
        assert resp.status_code == 200
        assert "PROMPTLINE" in resp.text

        # API routes still win over the static mount.
        assert client.get("/prompts/main/active").status_code == 404
        # GET /runs returns JSON (API), not the SPA, even with static mount active.
        assert client.get("/runs").json() == []
        cand = _cand("p1")
        registry.register(cand, "main")
        registry.activate("main", "p1")
        resp = client.get("/prompts/main/active")
        assert resp.status_code == 200
        assert resp.json()["prompt_id"] == "p1"


def test_no_web_dist_keeps_plain_404_at_root(tmp_path: Path) -> None:
    registry = PromptRegistry(tmp_path / "registry")
    run_manager = RunManager(tmp_path / "runs")
    # Directory without index.html → mount is skipped entirely.
    empty = tmp_path / "empty_dist"
    empty.mkdir()
    app = create_app(registry, run_manager, web_dist=empty)
    with TestClient(app) as client:
        assert client.get("/").status_code == 404
        assert client.get("/runs").json() == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# Judge-as-metric: POST /runs refuses uncalibrated judge with 400
# ---------------------------------------------------------------------------


def test_server_run_uncalibrated_judge_returns_400(tmp_path: Path) -> None:
    """POST /runs with judge metric enabled and no certificate → HTTP 400."""
    import json as _json
    import os

    import yaml

    from promptline.cli.main import build_app_from_config

    cfg_path = tmp_path / "promptline.yaml"
    cfg = {
        "program": {
            "name": "main",
            "instruction": "Answer.",
            "inputs": ["question"],
            "outputs": ["answer"],
        },
        "models": {"task": "fake/model", "reflection": "", "judge": ""},
        "dataset": {"kind": "jsonl", "path": ""},
        "judge": {"enabled": True},
        "budget": {"max_rollouts": 10, "max_cost_usd": None},
        "registry": {"path": str(tmp_path / "reg")},
    }
    cfg_path.write_text(yaml.dump(cfg))

    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        _json.dumps({"inputs": {"question": "q"}, "labels": {"answer": "a"}}) + "\n"
    )
    fake_path = tmp_path / "fake.json"
    fake_path.write_text(_json.dumps({"responses": ["[[answer]]: a"]}))

    env_backup = os.environ.copy()
    os.environ["PROMPTLINE_FAKE_SCRIPT"] = str(fake_path)
    try:
        server_app = build_app_from_config(str(cfg_path))
        with TestClient(server_app) as client:
            resp = client.post(
                "/runs",
                json={"optimizer": "bootstrap", "data_path": str(data_path)},
            )
            assert resp.status_code == 400, resp.text
            assert "calibrat" in resp.json()["detail"]
            # The failed run is inspectable.
            run_id = resp.json()["run_id"]
            assert client.get(f"/runs/{run_id}").json()["status"] == "failed"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


# ---------------------------------------------------------------------------
# Server gate parity: 400 on uncalibrated judge; promote activates winner
# ---------------------------------------------------------------------------


def test_gate_endpoint_uncalibrated_judge_400(tmp_path: Path) -> None:
    from promptline.judge.calibrator import UncalibratedJudgeError

    def _raise(payload: dict):
        raise UncalibratedJudgeError("no calibration certificate at cert.json")

    app, _, _ = _make_app(tmp_path, gate_runner=_raise)
    with TestClient(app) as client:
        resp = client.post(
            "/gate",
            json={"program": "main", "candidate_ids": ["p1"]},
        )
        assert resp.status_code == 400
        assert "certificate" in resp.json()["detail"]


def _gate_parity_fixture(tmp_path: Path):
    """Config + registry + splits + keyed fake so 'good-1' beats 'base-1'."""
    import os

    import yaml

    from promptline.registry.registry import PromptRegistry

    cfg_path = tmp_path / "promptline.yaml"
    registry_dir = tmp_path / "reg"
    cfg = {
        "program": {
            "name": "main",
            "instruction": "Answer the question.",
            "inputs": ["question"],
            "outputs": ["answer"],
        },
        "models": {"task": "fake/model", "reflection": "", "judge": ""},
        "dataset": {"kind": "jsonl", "path": ""},
        "judge": {"enabled": False},
        "budget": {"max_rollouts": 500, "max_cost_usd": None},
        "gate": {"alpha": 0.05, "min_examples": 20},
        "registry": {"path": str(registry_dir)},
    }
    cfg_path.write_text(yaml.dump(cfg))

    registry = PromptRegistry(registry_dir)
    registry.register(_cand("base-1", "base answer"), "main")
    registry.register(_cand("good-1", "good answer"), "main")
    registry.activate("main", "base-1")

    dev_path = tmp_path / "dev.jsonl"
    val_path = tmp_path / "val.jsonl"
    dev_path.write_text(
        "".join(
            json.dumps(
                {"inputs": {"question": f"d{i}"}, "labels": {"answer": "RIGHT"}}
            )
            + "\n"
            for i in range(25)
        )
    )
    val_path.write_text(
        "".join(
            json.dumps(
                {"inputs": {"question": f"v{i}"}, "labels": {"answer": "RIGHT"}}
            )
            + "\n"
            for i in range(12)
        )
    )

    fake_path = tmp_path / "fake.json"
    fake_path.write_text(
        json.dumps(
            {
                "keyed": [
                    {"contains": "good answer", "response": "[[answer]]: RIGHT"}
                ],
                "responses": ["[[answer]]: WRONG"],
            }
        )
    )
    env_backup = os.environ.copy()
    os.environ["PROMPTLINE_FAKE_SCRIPT"] = str(fake_path)
    return cfg_path, registry, dev_path, val_path, env_backup


def test_gate_endpoint_promotes_and_activates_winner(tmp_path: Path) -> None:
    """POST /gate promote verdict → winner activated, activated: true."""
    import os

    from promptline.cli.main import build_app_from_config

    cfg_path, registry, dev_path, val_path, env_backup = _gate_parity_fixture(
        tmp_path
    )
    try:
        server_app = build_app_from_config(str(cfg_path))
        with TestClient(server_app) as client:
            resp = client.post(
                "/gate",
                json={
                    "program": "main",
                    "candidate_ids": ["good-1"],
                    "dev_path": str(dev_path),
                    "val_path": str(val_path),
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["verdict"] == "promote"
            assert body["winner_id"] == "good-1"
            assert body["activated"] is True
            active = registry.get_active("main")
            assert active is not None and active[0] == "good-1"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_gate_endpoint_promote_false_does_not_activate(tmp_path: Path) -> None:
    """promote=false → verdict still reported but nothing is activated."""
    import os

    from promptline.cli.main import build_app_from_config

    cfg_path, registry, dev_path, val_path, env_backup = _gate_parity_fixture(
        tmp_path
    )
    try:
        server_app = build_app_from_config(str(cfg_path))
        with TestClient(server_app) as client:
            resp = client.post(
                "/gate",
                json={
                    "program": "main",
                    "candidate_ids": ["good-1"],
                    "dev_path": str(dev_path),
                    "val_path": str(val_path),
                    "promote": False,
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["verdict"] == "promote"
            assert body["activated"] is False
            active = registry.get_active("main")
            assert active is not None and active[0] == "base-1"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
