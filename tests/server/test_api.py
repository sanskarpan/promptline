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


def _fake_run_starter(spec, emit):
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
        assert resp.headers["etag"] == "p1"
        body = resp.json()
        assert body["program"] == "main"
        assert body["prompt_id"] == "p1"
        assert body["modules"]["main"]["instruction"] == "Be helpful."
        assert body["activated_at"]

        cached = client.get(
            "/prompts/main/active", headers={"If-None-Match": "p1"}
        )
        assert cached.status_code == 304
        assert not cached.content

        stale = client.get(
            "/prompts/main/active", headers={"If-None-Match": "other"}
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
    def _boom_starter(spec, emit):
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

    def factory(emit):
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
