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
    #: Back-compat only — prefer ``judge.certificate``, which is the primary
    #: location.  ``GateSettings.from_config`` falls back to
    #: ``judge.certificate`` when this is empty.
    certificate: str = ""
    #: Minimum kappa the certificate must attest when one is required.
    min_kappa: float = 0.6


class JudgeConfig(BaseModel):
    """LLM-judge metric configuration (the default optimization/gate metric)."""

    #: Rubric criterion name (also the certificate filename stem).
    criterion: str = "helpfulness"
    #: Rubric description; empty = use the built-in default for *criterion*.
    description: str = ""
    #: Integer rubric scale bounds.
    scale_min: int = 1
    scale_max: int = 5
    #: When true (default), optimize/gate score with the calibrated judge;
    #: when false they fall back to the exact-match metric.
    enabled: bool = True
    #: Path to the calibration certificate JSON.  Empty = the default
    #: ``<registry>/certificates/<criterion>.json`` written by
    #: ``promptline calibrate``.  This is the primary certificate location;
    #: ``gate.certificate`` is honored for back-compat when set.
    certificate: str = ""
    #: Minimum quadratic-weighted kappa the certificate must attest.
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
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when a config file cannot be parsed into a :class:`PromptlineConfig`.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers
    keep working; CLI callers catch it to print a clean message instead of a
    raw YAML traceback.
    """


def load_config(path: str | Path) -> PromptlineConfig:
    """Parse *path* (YAML) into a :class:`PromptlineConfig`.

    Raises :class:`ConfigError` (a ``ValueError``) with a readable message when
    the file is not valid YAML, rather than letting a raw ``yaml.YAMLError``
    escape as a traceback.
    """
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc
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

judge:
  # The calibrated LLM judge is the default optimize/gate metric.
  # Set enabled: false to fall back to exact-match on labels['answer'].
  enabled: true
  criterion: helpfulness
  # certificate: ""  # default: <registry>/certificates/<criterion>.json
  min_kappa: 0.6     # certificate must attest at least this kappa

gate:
  alpha: 0.05       # significance level for statistical gating
  min_examples: 50  # minimum eval examples required before deploying

registry:
  path: .promptline  # directory for run artefacts and the LLM response cache
"""
