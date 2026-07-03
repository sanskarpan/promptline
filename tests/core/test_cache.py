import pytest

from promptline.core.cache import CachingClient, LLMCache
from promptline.core.llm import FakeLLMClient, LLMCall, LLMResponse


def make_call(content: str = "hello") -> LLMCall:
    return LLMCall(model="gpt-4", messages=[{"role": "user", "content": content}])


def make_cache(tmp_path) -> LLMCache:
    return LLMCache(tmp_path / "test.db")


def test_cache_miss_returns_none(tmp_path):
    cache = make_cache(tmp_path)
    call = make_call()
    assert cache.get(call) is None


def test_put_then_get_returns_response(tmp_path):
    cache = make_cache(tmp_path)
    call = make_call()
    resp = LLMResponse(text="hello world")
    cache.put(call, resp)
    result = cache.get(call)
    assert result is not None
    assert result.text == "hello world"


def test_cache_hit_sets_cached_true(tmp_path):
    cache = make_cache(tmp_path)
    call = make_call()
    resp = LLMResponse(text="hi", cached=False)
    cache.put(call, resp)
    result = cache.get(call)
    assert result is not None
    assert result.cached is True


@pytest.mark.asyncio
async def test_caching_client_skips_inner_on_hit(tmp_path):
    cache = make_cache(tmp_path)
    fake = FakeLLMClient(script=["response1"])
    client = CachingClient(inner=fake, cache=cache)
    call = make_call()
    r1 = await client.complete(call)
    r2 = await client.complete(call)
    assert r1.text == "response1"
    assert r2.text == "response1"
    assert len(fake.calls) == 1  # inner called only once


def test_persistence(tmp_path):
    db_path = tmp_path / "persist.db"
    call = make_call("persistent")
    resp = LLMResponse(text="persistent value")

    cache1 = LLMCache(db_path)
    cache1.put(call, resp)
    cache1._conn.close()

    cache2 = LLMCache(db_path)
    result = cache2.get(call)
    assert result is not None
    assert result.text == "persistent value"
