from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass(frozen=True)
class Field:
    name: str
    desc: str = ""


@dataclass(frozen=True)
class Signature:
    instruction: str
    inputs: list[Field] = field(default_factory=list)
    outputs: list[Field] = field(default_factory=list)

    def render_system(self) -> str:
        lines = [self.instruction, "", "Inputs:"]
        for f in self.inputs:
            lines.append(f"- {f.name}: {f.desc}")
        lines.append("")
        lines.append("Outputs:")
        lines.append("Answer with each output field as [[name]]: value on its own section.")
        for f in self.outputs:
            lines.append(f"- [[{f.name}]]: {f.desc}")
        return "\n".join(lines)

    def parse_output(self, text: str) -> dict[str, str] | None:
        matches = re.findall(r"\[\[(\w+)\]\]:\s*(.*?)(?=\[\[|\Z)", text, re.DOTALL)
        if matches:
            return {k: v.strip() for k, v in matches}
        if len(self.outputs) == 1:
            return {self.outputs[0].name: text.strip()}
        return None


class Example(BaseModel):
    inputs: dict[str, str]
    labels: dict[str, str] = {}
    meta: dict = {}


class Demo(BaseModel):
    inputs: dict[str, str]
    outputs: dict[str, str]


class ModuleState(BaseModel):
    instruction: str
    demos: list[Demo] = []


class Candidate(BaseModel):
    id: str
    modules: dict[str, ModuleState]
    parent_ids: list[str] = []
    optimizer: str = ""
    meta: dict = {}

    @classmethod
    def seed(cls, modules: dict[str, ModuleState]) -> Candidate:
        return cls(id=uuid.uuid4().hex, modules=modules)

    def child(
        self,
        modules: dict[str, ModuleState],
        optimizer: str,
        extra_parents: list[str] = [],
    ) -> Candidate:
        return Candidate(
            id=uuid.uuid4().hex,
            modules=modules,
            parent_ids=[self.id, *extra_parents],
            optimizer=optimizer,
        )
