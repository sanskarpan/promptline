"""Promptline FastAPI application — serving plane + control plane.

Serving plane
    ``GET /prompts/{program}/active`` returns the active prompt with an ETag
    equal to the prompt id, so pollers can cheaply revalidate with
    ``If-None-Match`` (304 on match).

Control plane
    Run management (``/runs`` + SSE event streaming), gating (``/gate``),
    registry inspection/activation/rollback and judge certificates.

The app is dependency-injected: *run_starter* and *gate_runner* are closures
built by the CLI (``promptline serve``) from the project config; tests inject
fakes.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from sse_starlette.sse import EventSourceResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from promptline.judge.calibrator import UncalibratedJudgeError
from promptline.registry.registry import PromptRegistry
from promptline.server.runs import RunManager, RunStartError

#: Seconds between tail polls of a live run's events.jsonl.
_TAIL_POLL_S = 0.2


class _SPAStaticFiles(StaticFiles):
    """StaticFiles that serves ``index.html`` for unknown paths (SPA routing).

    Mounted at ``/`` *after* all API routes, so ``/runs`` etc. are still
    handled by the API; only paths no route matched reach this app.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise
        if response.status_code == 404:
            return await super().get_response("index.html", scope)
        return response


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class StartRunRequest(BaseModel):
    """Body of ``POST /runs``."""

    optimizer: str
    config_path: str = "promptline.yaml"
    data_path: str = ""
    budget: int | None = None


class GateRequest(BaseModel):
    """Body of ``POST /gate``; extra keys are passed through to gate_runner.

    ``dev_path`` and ``val_path`` are optional here (the real gate_runner
    requires them; missing values surface as a KeyError → 400).
    """

    model_config = ConfigDict(extra="allow")

    program: str
    incumbent_id: str = ""
    candidate_ids: list[str] = []
    dev_path: str = ""
    val_path: str = ""
    #: On a promote verdict, activate the winner (CLI parity).  The response
    #: reports the outcome in ``activated``.
    promote: bool = True


class ActivateRequest(BaseModel):
    prompt_id: str
    gate_report: dict = {}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    registry: PromptRegistry,
    run_manager: RunManager,
    run_starter: Callable | None = None,
    gate_runner: Callable | None = None,
    web_dist: Path | None = None,
) -> FastAPI:
    """Build the Promptline API.

    Parameters
    ----------
    registry:
        Prompt store backing the serving plane and registry endpoints.
    run_manager:
        Tracks optimizer runs; its ``base_dir`` holds each run's events.
    run_starter:
        ``(spec: StartRunRequest, emit) -> coroutine``; ``POST /runs``
        returns 400 when unset.
    gate_runner:
        ``(payload: dict) -> GateReport | dict`` (may be async);
        ``POST /gate`` returns 400 when unset.
    web_dist:
        Directory holding the built dashboard (``web/dist``).  When it
        contains an ``index.html`` it is mounted at ``/`` *after* the API
        routes, so API paths always win; unknown paths fall back to
        ``index.html`` (SPA routing).
    """
    app = FastAPI(title="promptline")

    # ---- Serving plane -----------------------------------------------------

    @app.get("/prompts/{program}/active")
    def get_active(program: str, request: Request) -> Response:
        info = registry.get_active_info(program)
        if info is None:
            raise HTTPException(404, f"no active prompt for program {program!r}")
        prompt_id: str = info["prompt_id"]
        # RFC 7232 §2.3: ETag field value must be a quoted-string.
        etag_value = f'"{prompt_id}"'
        # Strip surrounding quotes from the client's If-None-Match header to
        # compare the bare id; this handles both quoted and unquoted values.
        if_none_match = request.headers.get("if-none-match", "").strip('"')
        if if_none_match == prompt_id:
            return Response(status_code=304, headers={"ETag": etag_value})
        candidate = info["candidate"]
        body = {
            "program": program,
            "prompt_id": prompt_id,
            "modules": {name: state.model_dump() for name, state in candidate.modules.items()},
            "activated_at": info["activated_at"],
        }
        return Response(
            content=json.dumps(body),
            media_type="application/json",
            headers={"ETag": etag_value},
        )

    # ---- Control plane: runs -------------------------------------------------

    # response_model=None: FastAPI cannot build a response model from the
    # dict | JSONResponse union (the error path returns a JSONResponse).
    @app.post("/runs", response_model=None)
    async def start_run(spec: StartRunRequest) -> dict | JSONResponse:
        # async so RunManager.start creates the asyncio task on the serving loop.
        if run_starter is None:
            raise HTTPException(400, "run starting is not configured on this server")
        # Bind to a local so the None-narrowing survives into the lambda.
        starter = run_starter
        try:
            run_id = run_manager.start(lambda emit, run_dir: starter(spec, emit, run_dir))
        except RunStartError as exc:
            # Factory raised synchronously (e.g. dataset not found).  The run
            # is already stored as failed; return 400 with the error and
            # the run_id so the client can inspect GET /runs/{id}.
            return JSONResponse(
                status_code=400,
                content={"detail": str(exc.cause), "run_id": exc.run_id},
            )
        return {"run_id": run_id}

    @app.get("/runs")
    def list_runs() -> list[dict]:
        return run_manager.list()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        info = run_manager.get(run_id)
        if info is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        return info

    @app.get("/runs/{run_id}/events")
    def stream_events(run_id: str) -> EventSourceResponse:
        if run_manager.get(run_id) is None:
            raise HTTPException(404, f"unknown run {run_id!r}")

        async def _gen():
            path = run_manager.events_path(run_id)
            offset = 0

            def _drain() -> list[str]:
                nonlocal offset
                if not path.exists():
                    return []
                lines: list[str] = []
                with path.open("rb") as fh:
                    fh.seek(offset)
                    while True:
                        line = fh.readline()
                        if not line.endswith(b"\n"):
                            break  # EOF or partial write; retry next poll
                        offset = fh.tell()
                        stripped = line.decode().strip()
                        if stripped:
                            lines.append(stripped)
                return lines

            while True:
                info = run_manager.get(run_id)
                running = info is not None and info["status"] == "running"
                for data in _drain():
                    yield {"event": "run_event", "data": data}
                if not running:
                    break
                await asyncio.sleep(_TAIL_POLL_S)

        return EventSourceResponse(_gen())

    # ---- Control plane: gate ---------------------------------------------------

    @app.post("/gate")
    async def gate(payload: GateRequest) -> dict:
        if gate_runner is None:
            raise HTTPException(400, "gating is not configured on this server")
        try:
            result = gate_runner(payload.model_dump())
            if inspect.isawaitable(result):
                result = await result
        except UncalibratedJudgeError as exc:
            # Judge metric configured but no passing calibration certificate.
            raise HTTPException(400, str(exc)) from exc
        except (ValueError, KeyError, FileNotFoundError) as exc:
            # Covers: no incumbent, unknown candidate id, missing dev/val path.
            raise HTTPException(400, str(exc)) from exc
        if isinstance(result, BaseModel):
            return result.model_dump()
        return result

    # ---- Control plane: registry -------------------------------------------------

    @app.get("/registry/{program}")
    def registry_list(program: str) -> list[dict]:
        return registry.list_prompts(program)

    @app.post("/registry/{program}/activate")
    def registry_activate(program: str, body: ActivateRequest = Body(...)) -> dict:
        try:
            registry.activate(program, body.prompt_id, json.dumps(body.gate_report))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"program": program, "prompt_id": body.prompt_id}

    @app.post("/registry/{program}/rollback")
    def registry_rollback(program: str) -> dict:
        try:
            target = registry.rollback(program)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"program": program, "prompt_id": target}

    # ---- Control plane: judge certificates ------------------------------------------

    @app.get("/judges/certificates")
    def certificates() -> list[dict]:
        cert_dir = Path(registry.root) / "certificates"
        certs: list[dict] = []
        for path in sorted(cert_dir.glob("*.json")):
            certs.append(json.loads(path.read_text()))
        return certs

    # ---- Static dashboard (mounted last so API routes always win) --------------

    if web_dist is not None and (web_dist / "index.html").exists():
        app.mount("/", _SPAStaticFiles(directory=web_dist, html=True), name="web")

    return app
