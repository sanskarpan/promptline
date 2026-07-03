from __future__ import annotations

import asyncio
import json as _json
import os
import random
from collections.abc import Awaitable, Callable

import httpx

from promptline.core.llm import LLMCall, LLMError, LLMResponse

_SleepFn = Callable[[float], Awaitable[None]]


class OpenRouterClient:
    """Async HTTP client for the OpenRouter chat-completions API.

    Parameters
    ----------
    api_key:
        API key.  Falls back to the ``OPENROUTER_API_KEY`` environment
        variable.  ``LLMError`` is raised at construction time if neither
        is available.
    base_url:
        Root URL for the API (useful for pointing at a staging endpoint).
    max_retries:
        Number of *additional* attempts after the first failure on
        retriable status codes (429, 5xx) and network errors.  Total
        attempts = max_retries + 1.
    timeout:
        Per-request timeout in seconds passed to ``httpx``.
    _sleep:
        Injected sleep coroutine (replaces ``asyncio.sleep``).  Intended
        for test use only so that tests can run without real delays.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 4,
        timeout: float = 60.0,
        _sleep: _SleepFn | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMError(
                "No OpenRouter API key: pass api_key= or set OPENROUTER_API_KEY"
            )
        self._api_key: str = key
        self._base_url: str = base_url.rstrip("/")
        self._max_retries: int = max_retries
        self._timeout: float = timeout
        self._sleep: _SleepFn = _sleep if _sleep is not None else asyncio.sleep
        # Lazily-created, shared across complete() calls.
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared httpx client, creating it on first use."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def complete(self, call: LLMCall) -> LLMResponse:
        """Send *call* to OpenRouter and return the parsed response.

        Raises
        ------
        LLMError
            On unretriable HTTP errors (4xx ≠ 429), malformed 200
            responses (bad JSON or missing ``choices``), or once all
            retry attempts are exhausted.
        """
        body: dict = {
            "model": call.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in call.messages
            ],
            "temperature": call.temperature,
            "max_tokens": call.max_tokens,
            "usage": {"include": True},
        }
        if call.seed is not None:
            body["seed"] = call.seed

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: LLMError | None = None
        delay: float = 0.5
        attempt: int = 0

        http = self._get_client()
        while True:
            try:
                resp = await http.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        text: str = data["choices"][0]["message"]["content"]
                    except (_json.JSONDecodeError, KeyError, IndexError) as exc:
                        raise LLMError(
                            f"Malformed response from OpenRouter: {exc}"
                        ) from exc
                    usage: dict = data.get("usage", {})
                    return LLMResponse(
                        text=text,
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                        cost_usd=float(usage.get("cost", 0.0)),
                    )
                elif resp.status_code == 429 or resp.status_code >= 500:
                    last_error = LLMError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                else:
                    # 4xx other than 429 — no retry
                    raise LLMError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = LLMError(str(exc))

            # Decide whether to retry
            if attempt >= self._max_retries:
                raise last_error  # type: ignore[misc]

            jitter = random.uniform(0.0, delay * 0.1)
            await self._sleep(min(delay + jitter, 8.0))
            delay = min(delay * 2, 8.0)
            attempt += 1
