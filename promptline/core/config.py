"""Configuration schema and loader for Promptline.

Loads ``promptline.yaml`` into a :class:`PromptlineConfig` pydantic model.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ProgramConfig(BaseModel):
    name: str = "main"
    instruction: str = "Solve the task."
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class ModelsConfig(BaseModel):
    task: str = ""
    reflection: str = ""
    judge: str = ""


class DatasetConfig(BaseModel):
    kind: str = "jsonl"
    path: str = ""


class BudgetConfig(BaseModel):
    max_rollouts: int = 200
    max_cost_usd: float | None = None


class GateConfig(BaseModel):
    alpha: float = 0.05
    min_examples: int = 50
    #: Path to a judge calibration certificate JSON; empty = not required.
    certificate: str = ""
    #: Minimum kappa the certificate must attest when one is required.
    min_kappa: float = 0.6


class RegistryConfig(BaseModel):
    path: str = ".promptline"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class PromptlineConfig(BaseModel):
    program: ProgramConfig = Field(default_factory=ProgramConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> PromptlineConfig:
    """Parse *path* (YAML) into a :class:`PromptlineConfig`."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raw = {}
    return PromptlineConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Default config YAML
# ---------------------------------------------------------------------------


def default_config_yaml() -> str:
    """Return a commented sample ``promptline.yaml`` as a string."""
    return """\
# promptline.yaml — Promptline project configuration
# See https://github.com/your-org/promptline for full docs.

program:
  name: main
  # The zero-shot instruction for your task.
  instruction: "Answer the question."
  # Input field names (must match keys in your JSONL dataset's 'inputs' dict).
  inputs:
    - question
  # Output field names (must match the labels your metric checks).
  outputs:
    - answer

models:
  # Any model ID supported by OpenRouter (https://openrouter.ai/models).
  task: openai/gpt-4o-mini
  # Used for OPRO instruction proposals (leave blank to reuse task model).
  reflection: ""
  # Used for LLM-as-judge metrics (leave blank to reuse task model).
  judge: ""

dataset:
  kind: jsonl   # only 'jsonl' supported in the open-source build
  path: data.jsonl

budget:
  max_rollouts: 200
  # max_cost_usd: 5.0  # uncomment to add a USD cap

gate:
  alpha: 0.05       # significance level for statistical gating
  min_examples: 50  # minimum eval examples required before deploying

registry:
  path: .promptline  # directory for run artefacts and the LLM response cache
"""
