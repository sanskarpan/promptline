import pytest

from promptline.core.llm import FakeLLMClient, LLMCall, LLMError, Message


def make_call(content: str = "hello") -> LLMCall:
    return LLMCall(
        model="gpt-4",
        messages=[{"role": "user", "content": content}],
    )


@pytest.mark.asyncio
async def test_fake_script_pops_in_order():
    client = FakeLLMClient(script=["first", "second"])
    r1 = await client.complete(make_call())
    r2 = await client.complete(make_call())
    assert r1.text == "first"
    assert r2.text == "second"


@pytest.mark.asyncio
async def test_fake_records_calls():
    client = FakeLLMClient()
    await client.complete(make_call("hello"))
    await client.complete(make_call("world"))
    assert len(client.calls) == 2
    assert client.calls[0].messages[0].content == "hello"


@pytest.mark.asyncio
async def test_fake_callable_mode():
    client = FakeLLMClient(script=lambda call: f"echo:{call.messages[0].content}")
    r = await client.complete(make_call("test"))
    assert r.text == "echo:test"


def test_key_identical_for_same_content():
    call1 = LLMCall(model="gpt-4", messages=[{"role": "user", "content": "hello"}])
    call2 = LLMCall(model="gpt-4", messages=[Message(role="user", content="hello")])
    assert call1.key() == call2.key()


def test_key_differs_for_different_content():
    call1 = LLMCall(model="gpt-4", messages=[{"role": "user", "content": "hello"}])
    call2 = LLMCall(model="gpt-4", messages=[{"role": "user", "content": "world"}])
    assert call1.key() != call2.key()


@pytest.mark.asyncio
async def test_fake_script_exhausted_raises():
    client = FakeLLMClient(script=[])
    with pytest.raises(LLMError):
        await client.complete(make_call())
