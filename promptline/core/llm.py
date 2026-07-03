from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: str
    content: str


class LLMCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def _normalize_messages(cls, v: object) -> tuple[Message, ...]:
        result = []
        for m in v:  # type: ignore[union-attr]
            if isinstance(m, dict):
                result.append(Message(**m))
            else:
                result.append(m)
        return tuple(result)

    def key(self) -> str:
        data = self.model_dump()
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=list).encode()).hexdigest()


class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    async def complete(self, call: LLMCall) -> LLMResponse: ...


class FakeLLMClient:
    def __init__(
        self,
        script: list[str] | Callable[[LLMCall], str] | None = None,
    ):
        self.script = script
        self.calls: list[LLMCall] = []

    async def complete(self, call: LLMCall) -> LLMResponse:
        self.calls.append(call)
        if self.script is None:
            return LLMResponse(text="FAKE")
        elif callable(self.script):
            return LLMResponse(text=self.script(call))
        else:
            if not self.script:
                raise LLMError("FakeLLMClient script exhausted")
            return LLMResponse(text=self.script.pop(0))
