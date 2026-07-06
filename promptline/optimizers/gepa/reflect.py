"""Reflective mutation for GEPA: prompt construction and reply parsing.

The reflection LLM sees the target module's current instruction plus, for each
minibatch example, the module's inputs, its raw output, the score and the
textual feedback from the metric.  It must reply with a new instruction inside
a fenced code block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DIRECTIVE = (
    "Diagnose the failures in the examples above, then write an improved "
    "instruction for this module. Output ONLY the new instruction inside a "
    "fenced code block."
)

_FENCED_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


@dataclass
class ReflectionExample:
    """One minibatch example as seen by the reflection model."""

    inputs: str
    output: str
    score: float
    feedback: str


def build_reflection_prompt(
    module_name: str,
    instruction: str,
    examples: list[ReflectionExample],
) -> str:
    """Build the reflection prompt for module *module_name*."""
    lines: list[str] = [
        f'You are improving the instruction of the module "{module_name}" '
        "inside a multi-module LLM program.",
        "",
        "Current instruction:",
        "```",
        instruction,
        "```",
        "",
        "Here is how the module performed on a minibatch of examples:",
    ]
    for i, ex in enumerate(examples, start=1):
        lines += [
            "",
            f"### Example {i}",
            "Module inputs:",
            ex.inputs,
            "Module output:",
            ex.output,
            f"Score: {ex.score:.4f}",
            f"Feedback: {ex.feedback}",
        ]
    lines += ["", DIRECTIVE]
    return "\n".join(lines)


def parse_new_instruction(text: str) -> str:
    """Extract the first fenced code block; fall back to the stripped reply."""
    match = _FENCED_RE.search(text)
    if match:
        block = match.group(1).strip()
        if block:
            return block
    return text.strip()
