from __future__ import annotations

import json

import httpx
import pytest
import respx

from promptline.core.llm import LLMCall, LLMError, Message
from promptline.core.openrouter import OpenRouterClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call(**kwargs) -> LLMCall:
    return LLMCall(
        model="openai/gpt-4o",
        messages=(Message(role="user", content="Hello"),),
        temperature=0.2,
        max_tokens=100,
        **kwargs,
    )


def _ok_response(
    text: str = "Hi", prompt: int = 10, completion: int = 5, cost: float = 0.001
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cost": cost,
            },
        },
    )


async def _no_sleep(_: float) -> None:
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_maps_text_tokens_cost() -> None:
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=_ok_response("Hello world", prompt=12, completion=7, cost=0.002)
        )
        client = OpenRouterClient(api_key="test-key")
        resp = await client.complete(_make_call())

    assert resp.text == "Hello world"
    assert resp.prompt_tokens == 12
    assert resp.completion_tokens == 7
    assert resp.cost_usd == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_cost_absent_defaults_to_zero() -> None:
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "text"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            )
        )
        client = OpenRouterClient(api_key="test-key")
        resp = await client.complete(_make_call())

    assert resp.cost_usd == 0.0


@pytest.mark.asyncio
async def test_429_then_200_succeeds_with_two_requests() -> None:
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/chat/completions")
        route.side_effect = [
            httpx.Response(429, text="rate limited"),
            _ok_response("Recovered"),
        ]
        client = OpenRouterClient(api_key="test-key", _sleep=_no_sleep)
        resp = await client.complete(_make_call())

    assert resp.text == "Recovered"
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_500_five_times_raises_llm_error() -> None:
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/chat/completions")
        route.side_effect = [httpx.Response(500, text="internal server error")] * 5
        client = OpenRouterClient(api_key="test-key", max_retries=4, _sleep=_no_sleep)
        with pytest.raises(LLMError, match="500"):
            await client.complete(_make_call())

    assert route.call_count == 5


@pytest.mark.asyncio
async def test_401_raises_immediately_without_retry() -> None:
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/chat/completions")
        route.mock(return_value=httpx.Response(401, text="unauthorized"))
        client = OpenRouterClient(api_key="test-key", _sleep=_no_sleep)
        with pytest.raises(LLMError, match="401"):
            await client.complete(_make_call())

    assert route.call_count == 1


@pytest.mark.asyncio
async def test_auth_header_present() -> None:
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=_ok_response()
        )
        client = OpenRouterClient(api_key="my-secret-key")
        await client.complete(_make_call())

    req = route.calls[0].request
    assert req.headers["Authorization"] == "Bearer my-secret-key"


@pytest.mark.asyncio
async def test_seed_included_in_body_when_set() -> None:
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=_ok_response()
        )
        client = OpenRouterClient(api_key="test-key")
        await client.complete(_make_call(seed=42))

    body = json.loads(route.calls[0].request.content)
    assert body["seed"] == 42


@pytest.mark.asyncio
async def test_seed_not_in_body_when_not_set() -> None:
    with respx.mock:
        route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=_ok_response()
        )
        client = OpenRouterClient(api_key="test-key")
        await client.complete(_make_call())

    body = json.loads(route.calls[0].request.content)
    assert "seed" not in body


def test_no_api_key_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError):
        OpenRouterClient()


def test_env_api_key_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    client = OpenRouterClient()
    assert client._api_key == "env-key"


# ---------------------------------------------------------------------------
# Connection pooling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pooled_client_lazily_created() -> None:
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=_ok_response()
        )
        client = OpenRouterClient(api_key="test-key")
        assert client._http is None
        await client.complete(_make_call())
        assert client._http is not None


@pytest.mark.asyncio
async def test_pooled_client_reused_across_calls() -> None:
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=_ok_response()
        )
        client = OpenRouterClient(api_key="test-key")
        await client.complete(_make_call())
        first_http = client._http
        await client.complete(_make_call())
        assert client._http is first_http


@pytest.mark.asyncio
async def test_aclose_resets_pooled_client() -> None:
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=_ok_response()
        )
        client = OpenRouterClient(api_key="test-key")
        await client.complete(_make_call())
        assert client._http is not None
        await client.aclose()
        assert client._http is None


@pytest.mark.asyncio
async def test_context_manager_closes_on_exit() -> None:
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=_ok_response("context-result")
        )
        async with OpenRouterClient(api_key="test-key") as client:
            resp = await client.complete(_make_call())
            assert resp.text == "context-result"
        assert client._http is None


# ---------------------------------------------------------------------------
# Malformed 200 responses → LLMError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_json_body_raises_llm_error() -> None:
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text="not json at all",
                headers={"content-type": "application/json"},
            )
        )
        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(LLMError, match="[Mm]alformed"):
            await client.complete(_make_call())


@pytest.mark.asyncio
async def test_missing_choices_key_raises_llm_error() -> None:
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"usage": {"prompt_tokens": 5}})
        )
        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(LLMError, match="[Mm]alformed"):
            await client.complete(_make_call())


@pytest.mark.asyncio
async def test_empty_choices_list_raises_llm_error() -> None:
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [], "usage": {}})
        )
        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(LLMError, match="[Mm]alformed"):
            await client.complete(_make_call())
