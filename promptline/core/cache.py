from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from promptline.core.llm import LLMCall, LLMClient, LLMResponse


class LLMCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, call: LLMCall) -> LLMResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM calls WHERE key = ?",
                (call.key(),),
            ).fetchone()
        if row is None:
            return None
        resp = LLMResponse.model_validate_json(row[0])
        return resp.model_copy(update={"cached": True})

    def put(self, call: LLMCall, resp: LLMResponse) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO calls (key, response_json, created_at) VALUES (?, ?, ?)",
                (call.key(), resp.model_dump_json(), datetime.now(UTC).isoformat()),
            )
            self._conn.commit()


class CachingClient:
    def __init__(self, inner: LLMClient, cache: LLMCache) -> None:
        self._inner = inner
        self._cache = cache

    async def complete(self, call: LLMCall) -> LLMResponse:
        hit = self._cache.get(call)
        if hit is not None:
            return hit
        resp = await self._inner.complete(call)
        self._cache.put(call, resp)
        return resp
