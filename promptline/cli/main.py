"""Promptline command-line interface.

Entry point: ``promptline`` (see ``[project.scripts]`` in pyproject.toml).
"""
from __future__ import annotations

import asyncio
import json
import os
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from promptline import __version__
from promptline.core.config import PromptlineConfig, default_config_yaml, load_config
from promptline.core.llm import FakeLLMClient
from promptline.core.program import ModelConfig, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.data.dataset import Dataset
from promptline.eval.harness import Budget, EvalHarness, MetricResult
from promptline.judge.calibrator import Calibrator
from promptline.judge.judge import PointwiseJudge, RubricCriterion

console = Console()

app = typer.Typer(
    name="promptline",
    help="Prompt Optimization Pipeline CLI.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OptimizerChoice(StrEnum):
    bootstrap = "bootstrap"
    bootstrap_rs = "bootstrap-rs"
    opro = "opro"


def load_examples_jsonl(path: str) -> list[Example]:
    """Load examples from a JSONL file.

    Each line must be a JSON object with at least an ``inputs`` key.
    An optional ``labels`` key supplies ground-truth values.
    """
    examples: list[Example] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        examples.append(
            Example(
                inputs=obj.get("inputs", {}),
                labels=obj.get("labels", {}),
                meta=obj.get("meta", {}),
            )
        )
    return examples


def default_metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
    """Exact-match on ``labels['answer']`` vs ``outputs.get('answer')``.

    TODO: replace with an LLM-judge metric in a later task.
    """
    expected = example.labels.get("answer", "")
    got = prediction.outputs.get("answer", "")
    score = 1.0 if got.strip() == expected.strip() else 0.0
    return MetricResult(score=score, feedback=f"expected={expected!r} got={got!r}")


def _build_client(cfg: PromptlineConfig):  # type: ignore[return]
    """Return an LLM client.

    Uses :class:`FakeLLMClient` when ``PROMPTLINE_FAKE_SCRIPT`` env var is set
    (value = path to a JSON file ``{"responses": [...]}``, used as a cycling
    list).  Otherwise returns an ``OpenRouterClient`` wrapped in
    ``CachingClient``.
    """
    fake_script_path = os.environ.get("PROMPTLINE_FAKE_SCRIPT")
    if fake_script_path:
        data = json.loads(Path(fake_script_path).read_text())
        responses: list[str] = data["responses"]
        idx_state = {"i": 0}

        def _cyclic(call):  # noqa: ARG001
            text = responses[idx_state["i"] % len(responses)]
            idx_state["i"] += 1
            return text

        return FakeLLMClient(script=_cyclic)

    # Real path: OpenRouter + disk cache.
    from promptline.core.cache import CachingClient, LLMCache
    from promptline.core.openrouter import OpenRouterClient

    registry_path = Path(cfg.registry.path)
    registry_path.mkdir(parents=True, exist_ok=True)
    cache = LLMCache(registry_path / "cache.db")
    inner = OpenRouterClient()
    return CachingClient(inner=inner, cache=cache)


def _build_optimizer(choice: OptimizerChoice):
    if choice == OptimizerChoice.bootstrap:
        from promptline.optimizers.bootstrap import BootstrapFewShot
        return BootstrapFewShot()
    elif choice == OptimizerChoice.bootstrap_rs:
        from promptline.optimizers.bootstrap import BootstrapRandomSearch
        return BootstrapRandomSearch()
    else:
        from promptline.optimizers.opro import OPRO
        return OPRO()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing config."),
) -> None:
    """Write a starter promptline.yaml in the current directory."""
    config_path = Path("promptline.yaml")
    if config_path.exists() and not force:
        console.print(
            "[red]promptline.yaml already exists.[/red] "
            "Pass --force to overwrite."
        )
        raise typer.Exit(1)
    config_path.write_text(default_config_yaml())
    console.print(f"[green]Created[/green] {config_path.resolve()}")
    console.print("\nNext steps:")
    console.print("  1. Edit [bold]promptline.yaml[/bold] — set your instruction, inputs, outputs.")
    console.print(
        "  2. Prepare a [bold]data.jsonl[/bold] with "
        '{\"inputs\": {...}, \"labels\": {...}} lines.'
    )
    console.print("  3. Run [bold]promptline optimize[/bold] to start optimizing.")


@app.command()
def version() -> None:
    """Print the installed Promptline version."""
    console.print(__version__)


@app.command()
def optimize(
    optimizer: OptimizerChoice = typer.Option(
        OptimizerChoice.bootstrap,
        "--optimizer",
        help="Which optimizer to run.",
    ),
    config: str = typer.Option(
        "promptline.yaml",
        "--config",
        help="Path to promptline.yaml.",
    ),
    budget: int | None = typer.Option(
        None,
        "--budget",
        help="Override max_rollouts from config.",
    ),
    data: str | None = typer.Option(
        None,
        "--data",
        help="Path to JSONL dataset (overrides config dataset.path).",
    ),
) -> None:
    """Run a prompt optimization pass and print the best candidate."""

    # ---- Load config --------------------------------------------------------
    cfg_path = Path(config)
    if not cfg_path.exists():
        console.print(f"[red]Config not found:[/red] {cfg_path}")
        raise typer.Exit(1)
    cfg = load_config(cfg_path)

    # ---- Dataset ------------------------------------------------------------
    data_path = data or cfg.dataset.path
    if not data_path:
        console.print(
            "[red]No dataset path specified.[/red] "
            "Use --data or set dataset.path in config."
        )
        raise typer.Exit(1)
    if not Path(data_path).exists():
        console.print(f"[red]Dataset not found:[/red] {data_path}")
        raise typer.Exit(1)
    examples = load_examples_jsonl(data_path)
    if not examples:
        console.print("[red]Dataset is empty.[/red]")
        raise typer.Exit(1)

    # ---- Program & seed -----------------------------------------------------
    program = PromptProgram.simple(
        instruction=cfg.program.instruction,
        inputs=cfg.program.inputs,
        outputs=cfg.program.outputs,
        name=cfg.program.name,
    )
    seed = Candidate.seed(
        modules={
            cfg.program.name: ModuleState(instruction=cfg.program.instruction)
        }
    )

    # ---- Budget -------------------------------------------------------------
    max_rollouts = budget if budget is not None else cfg.budget.max_rollouts
    run_budget = Budget(
        max_rollouts=max_rollouts,
        max_cost_usd=cfg.budget.max_cost_usd,
    )

    # ---- Model config -------------------------------------------------------
    model_cfg = ModelConfig(
        task_model=cfg.models.task or "openai/gpt-4o-mini",
        reflection_model=cfg.models.reflection,
        judge_model=cfg.models.judge,
    )

    # ---- Client & harness ---------------------------------------------------
    client = _build_client(cfg)
    harness = EvalHarness(client=client, cfg=model_cfg)

    # ---- Optimizer ----------------------------------------------------------
    opt = _build_optimizer(optimizer)

    collected_events: list = []

    def _emit(event):
        collected_events.append(event)

    async def _run():
        return await opt.optimize(
            program=program,
            seed=seed,
            trainset=examples,
            metric=default_metric,
            budget=run_budget,
            harness=harness,
            emit=_emit,
        )

    console.print(
        f"\nRunning [bold]{optimizer.value}[/bold] optimizer "
        f"on {len(examples)} examples …"
    )
    result = asyncio.run(_run())

    # ---- Results table ------------------------------------------------------
    table = Table(title="Optimization Results", show_lines=True)
    table.add_column("Candidate ID", style="dim", max_width=12)
    table.add_column("Score", justify="right")
    table.add_column("Instruction (excerpt)", max_width=60)

    # Sort candidates by score descending.
    scored = [
        (c, result.scores.get(c.id, float("nan")))
        for c in result.candidates
    ]
    scored.sort(key=lambda x: x[1] if x[1] == x[1] else -1, reverse=True)

    for cand, score in scored[:10]:
        first_mod = next(iter(cand.modules.values()))
        instr_excerpt = first_mod.instruction[:80].replace("\n", " ")
        score_str = f"{score:.3f}" if score == score else "—"
        table.add_row(cand.id[:10], score_str, instr_excerpt)

    console.print(table)

    # Best instruction panel.
    best_first_mod = next(iter(result.best.modules.values()))
    _best_raw = result.scores.get(result.best.id)
    _best_score_str = f"{_best_raw:.3f}" if isinstance(_best_raw, float) else "n/a"
    console.print(
        Panel(
            best_first_mod.instruction,
            title=f"Best Instruction (score={_best_score_str})",
            border_style="green",
        )
    )
    console.print(f"\n[bold]Rollouts used:[/bold] {run_budget.rollouts_used}")
    console.print(f"[bold]Events emitted:[/bold] {result.events_count}")


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

#: Default rubric descriptions per criterion name.
DEFAULT_RUBRICS: dict[str, str] = {
    "helpfulness": (
        "How well the response addresses the user's request and provides "
        "actionable, relevant information."
    ),
    "correctness": (
        "Whether the response is factually accurate and free of errors or "
        "unsupported claims."
    ),
    "coherence": (
        "Whether the response is well-structured, consistent, and easy to follow."
    ),
    "complexity": (
        "The intellectual depth required to write the response "
        "(domain expertise vs. basic language competency)."
    ),
    "verbosity": (
        "Whether the amount of detail is appropriate for the request — "
        "neither too terse nor padded."
    ),
}


def _load_gold_dataset(gold: str, criterion: str, n: int | None) -> Dataset:
    """Load the gold dataset from a JSONL path or the HelpSteer2 HF loader."""
    if gold == "helpsteer2":
        from promptline.data.loaders import load_helpsteer2

        attribute = criterion if criterion in DEFAULT_RUBRICS else "helpfulness"
        return load_helpsteer2(attribute=attribute, limit=n)
    gold_path = Path(gold)
    if not gold_path.exists():
        console.print(f"[red]Gold dataset not found:[/red] {gold_path}")
        raise typer.Exit(1)
    dataset = Dataset.from_jsonl(gold_path)
    if n is not None:
        dataset = Dataset(dataset.records[:n])
    return dataset


@app.command()
def calibrate(
    gold: str = typer.Option(
        ...,
        "--gold",
        help="Path to a gold JSONL dataset, or the literal 'helpsteer2'.",
    ),
    criterion: str = typer.Option(
        "helpfulness",
        "--criterion",
        help="Rubric criterion to calibrate the judge on.",
    ),
    n: int = typer.Option(200, "--n", help="Max gold records to use."),
    threshold: float = typer.Option(
        0.6,
        "--threshold",
        help="Minimum quadratic-weighted kappa required to pass.",
    ),
    config: str = typer.Option(
        "promptline.yaml",
        "--config",
        help="Path to promptline.yaml.",
    ),
) -> None:
    """Calibrate the LLM judge against gold human labels and save a certificate."""

    # ---- Config ------------------------------------------------------------
    cfg_path = Path(config)
    cfg = load_config(cfg_path) if cfg_path.exists() else PromptlineConfig()

    # ---- Gold dataset --------------------------------------------------------
    dataset = _load_gold_dataset(gold, criterion, n)
    if not len(dataset):
        console.print("[red]Gold dataset is empty.[/red]")
        raise typer.Exit(1)

    # ---- Judge ---------------------------------------------------------------
    judge_model = cfg.models.judge or cfg.models.task or "openai/gpt-4o-mini"
    rubric = RubricCriterion(
        name=criterion,
        description=DEFAULT_RUBRICS.get(
            criterion, f"Rate the overall {criterion} of the response."
        ),
    )
    judge = PointwiseJudge(criterion=rubric, judge_model=judge_model)

    # ---- Calibrate -------------------------------------------------------------
    client = _build_client(cfg)
    calibrator = Calibrator(judge, dataset, client, threshold_kappa=threshold)

    console.print(
        f"\nCalibrating judge [bold]{judge_model}[/bold] on criterion "
        f"[bold]{criterion}[/bold] ({len(dataset)} gold records, "
        f"{len(calibrator.holdout)} holdout) …"
    )
    cert = asyncio.run(calibrator.calibrate())

    # ---- Report -----------------------------------------------------------------
    table = Table(title="Calibration Certificate")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("criterion", cert.criterion)
    table.add_row("kappa (quadratic)", f"{cert.kappa:.3f}")
    table.add_row("spearman", f"{cert.spearman:.3f}")
    table.add_row("n_holdout", str(cert.n_holdout))
    table.add_row("threshold", f"{cert.threshold:.2f}")
    table.add_row("binning", cert.binning)
    table.add_row(
        "passed",
        "[green]yes[/green]" if cert.passed else "[red]no[/red]",
    )
    console.print(table)

    # ---- Save ---------------------------------------------------------------------
    cert_path = Path(cfg.registry.path) / "certificates" / f"{criterion}.json"
    cert.save(cert_path)
    console.print(f"Certificate saved to [bold]{cert_path}[/bold]")

    if not cert.passed:
        console.print(
            f"[red]Calibration failed:[/red] kappa {cert.kappa:.3f} "
            f"< threshold {cert.threshold:.2f}"
        )
        raise typer.Exit(1)
