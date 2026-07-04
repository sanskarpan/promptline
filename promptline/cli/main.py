"""Promptline command-line interface.

Entry point: ``promptline`` (see ``[project.scripts]`` in pyproject.toml).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from promptline import __version__
from promptline.core.config import PromptlineConfig, default_config_yaml, load_config
from promptline.core.llm import FakeLLMClient, LLMError
from promptline.core.program import ModelConfig, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.data.dataset import Dataset
from promptline.eval.harness import Budget, EvalHarness
from promptline.judge.calibrator import (
    Calibrator,
    UncalibratedJudgeError,
    require_certificate,
)
from promptline.judge.judge import PointwiseJudge, RubricCriterion
from promptline.judge.metric_factory import (
    DEFAULT_RUBRICS,
    default_metric,  # noqa: F401 — re-exported: the exact-match fallback metric
    resolve_certificate_path,
    resolve_metric,
)
from promptline.optimizers.base import RunRecorder
from promptline.registry.registry import PromptRegistry

console = Console()

app = typer.Typer(
    name="promptline",
    help="Prompt Optimization Pipeline CLI.",
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit(0)


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the Promptline version and exit.",
    ),
) -> None:
    """Prompt Optimization Pipeline CLI."""

# ---------------------------------------------------------------------------
# demo + data sub-apps
# ---------------------------------------------------------------------------

from promptline.cli.demo import demo_app, demo_setup  # noqa: E402

app.add_typer(demo_app, name="demo")

data_app = typer.Typer(name="data", help="Data preparation utilities.")
app.add_typer(data_app, name="data")


@data_app.command("prepare")
def data_prepare(
    demo: bool = typer.Option(False, "--demo", help="Prepare demo data."),
    dir: str = typer.Option(
        "examples/support-assistant/workspace",
        "--dir",
        help="Workspace directory (forwarded to `demo setup`).",
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Use bundled fixtures (forwarded to `demo setup`)."
    ),
) -> None:
    """Prepare data for a Promptline pipeline.

    ``--demo`` is an alias that forwards to ``promptline demo setup``.
    """
    if demo:
        # Direct function call: pass explicit values (typer defaults are
        # OptionInfo sentinels when bypassing the CLI layer).
        demo_setup(dir=dir, offline=offline, gold_n=400, dev_n=150, val_n=150)
    raise typer.Exit(0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OptimizerChoice(StrEnum):
    bootstrap = "bootstrap"
    bootstrap_rs = "bootstrap-rs"
    opro = "opro"
    gepa = "gepa"
    protegi = "protegi"
    mipro = "mipro"


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
        responses: list[str] = data.get("responses", [])
        #: Optional prompt-keyed rules: [{"contains": ..., "response": ...}].
        keyed: list[dict] = data.get("keyed", [])
        idx_state = {"i": 0}

        def _scripted(call):
            blob = "\n".join(m.content for m in call.messages)
            for rule in keyed:
                if rule["contains"] in blob:
                    return rule["response"]
            if not responses:
                return ""
            text = responses[idx_state["i"] % len(responses)]
            idx_state["i"] += 1
            return text

        return FakeLLMClient(script=_scripted)

    # Real path: OpenRouter + disk cache.
    from promptline.core.cache import CachingClient, LLMCache
    from promptline.core.openrouter import OpenRouterClient

    registry_path = Path(cfg.registry.path)
    registry_path.mkdir(parents=True, exist_ok=True)
    cache = LLMCache(registry_path / "cache.db")
    inner = OpenRouterClient()
    return CachingClient(inner=inner, cache=cache)


def _build_optimizer(
    choice: OptimizerChoice,
    run_dir: Path | None = None,
    resume: bool = False,
):
    """Construct the chosen optimizer.

    GEPA persists its own events/checkpoints when *run_dir* is set (and
    resumes from it when *resume* is true); the other optimizers rely on the
    caller wiring a :class:`RunRecorder` into ``emit``.
    """
    if resume and choice != OptimizerChoice.gepa:
        raise ValueError("--resume is only supported for the gepa optimizer")
    if choice == OptimizerChoice.bootstrap:
        from promptline.optimizers.bootstrap import BootstrapFewShot
        return BootstrapFewShot()
    elif choice == OptimizerChoice.bootstrap_rs:
        from promptline.optimizers.bootstrap import BootstrapRandomSearch
        return BootstrapRandomSearch()
    elif choice == OptimizerChoice.gepa:
        from promptline.optimizers.gepa import GEPA
        return GEPA(run_dir=run_dir, resume_from=run_dir if resume else None)
    elif choice == OptimizerChoice.protegi:
        from promptline.optimizers.protegi import ProTeGi
        return ProTeGi()
    elif choice == OptimizerChoice.mipro:
        from promptline.optimizers.mipro import MIPRO
        return MIPRO()
    else:
        from promptline.optimizers.opro import OPRO
        return OPRO()


def _build_program_and_seed(cfg: PromptlineConfig) -> tuple[PromptProgram, Candidate]:
    program = PromptProgram.simple(
        instruction=cfg.program.instruction,
        inputs=cfg.program.inputs,
        outputs=cfg.program.outputs,
        name=cfg.program.name,
    )
    seed = Candidate.seed(
        modules={cfg.program.name: ModuleState(instruction=cfg.program.instruction)}
    )
    return program, seed


def _model_config(cfg: PromptlineConfig) -> ModelConfig:
    return ModelConfig(
        task_model=cfg.models.task or "openai/gpt-4o-mini",
        reflection_model=cfg.models.reflection,
        judge_model=cfg.models.judge,
    )


def _dataset_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _metric_label(cfg: PromptlineConfig, mode: str) -> str:
    """Human-readable metric description, e.g. ``judge(helpfulness)``."""
    return f"judge({cfg.judge.criterion})" if mode == "judge" else mode


def _build_run_coro(
    cfg: PromptlineConfig,
    optimizer_choice: OptimizerChoice,
    data_path: str,
    budget: Budget,
    emit,
    run_dir: Path,
    run_id: str,
    registry: PromptRegistry,
    resume: bool = False,
    allow_uncalibrated: bool = False,
) -> tuple:
    """Return ``(coroutine, metric_label)`` for one optimization pass.

    All synchronous setup (loading examples, constructing the LLM client and
    harness, resolving the metric, checking the judge certificate, building
    the optimizer) is performed here so that errors surface immediately to the
    caller.  On the server side :class:`RunManager` wraps this in a try/except
    and converts synchronous failures into ``RunStartError`` → HTTP 400 rather
    than leaving a zombie run.

    When the resolved metric is the LLM judge, a passing calibration
    certificate is required (``cfg.judge.certificate`` or the default
    ``<registry>/certificates/<criterion>.json``); a missing/weak certificate
    raises :class:`UncalibratedJudgeError` unless *allow_uncalibrated* is set.

    NOTE: when called from the gate sub-path, pass
    ``budget=Budget(max_rollouts=None, max_cost_usd=cfg.budget.max_cost_usd)``
    — gate runs are bounded by data size, not by rollout count; only the cost
    ceiling is honored there.
    """
    examples = load_examples_jsonl(data_path)
    program, seed = _build_program_and_seed(cfg)
    client = _build_client(cfg)
    harness = EvalHarness(client=client, cfg=_model_config(cfg))
    metric, metric_mode = resolve_metric(cfg, client)
    if metric_mode == "judge" and not allow_uncalibrated:
        # The spec's promise: optimization refuses to run against an
        # uncalibrated judge.
        require_certificate(resolve_certificate_path(cfg), cfg.judge.min_kappa)
    opt = _build_optimizer(optimizer_choice, run_dir=run_dir, resume=resume)

    async def _run():
        result = await opt.optimize(
            program=program,
            seed=seed,
            trainset=examples,
            metric=metric,
            budget=budget,
            harness=harness,
            emit=emit,
        )
        registry.register(result.best, cfg.program.name, run_id=run_id)
        best_score = result.scores.get(result.best.id)
        if isinstance(best_score, float) and best_score == best_score:
            registry.record_eval(
                result.best.id,
                dataset_hash=_dataset_hash(data_path),
                mean_score=best_score,
                n=len(examples),
            )
        return result

    return _run(), _metric_label(cfg, metric_mode)


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
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="Resume a previous run by id (gepa only).",
    ),
    allow_uncalibrated: bool = typer.Option(
        False,
        "--allow-uncalibrated",
        help="Run with the judge metric even without a passing calibration "
        "certificate (NOT recommended).",
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
    # Validate + count examples before kicking off the optimizer.
    try:
        examples_preview = load_examples_jsonl(data_path)
    except json.JSONDecodeError as exc:
        typer.secho(f"Dataset parse error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    if not examples_preview:
        console.print("[red]Dataset is empty.[/red]")
        raise typer.Exit(1)

    # ---- Budget -------------------------------------------------------------
    max_rollouts = budget if budget is not None else cfg.budget.max_rollouts
    run_budget = Budget(
        max_rollouts=max_rollouts,
        max_cost_usd=cfg.budget.max_cost_usd,
    )

    # ---- Run dir & emit setup -----------------------------------------------
    run_id = resume or uuid.uuid4().hex
    run_dir = Path(cfg.registry.path) / "runs" / run_id

    # GEPA writes events/checkpoints to run_dir itself; for the other
    # optimizers the CLI-owned recorder captures the event stream.
    if optimizer == OptimizerChoice.gepa:
        def _emit(event):  # GEPA's internal recorder already persists events.
            pass
    else:
        recorder = RunRecorder(run_dir)
        _emit = recorder.emit

    registry = PromptRegistry(Path(cfg.registry.path))

    # ---- Build run coroutine (synchronous setup; errors → exit 1/2) ---------
    try:
        coro, metric_label = _build_run_coro(
            cfg,
            optimizer,
            data_path,
            run_budget,
            _emit,
            run_dir,
            run_id,
            registry,
            resume=resume is not None,
            allow_uncalibrated=allow_uncalibrated,
        )
    except ValueError as exc:
        # _build_optimizer raises ValueError for invalid --resume combinations.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except UncalibratedJudgeError as exc:
        console.print(
            f"[red]judge not calibrated[/red] — {exc}\n"
            "Run: [bold]promptline calibrate --gold <gold.jsonl>[/bold] "
            "(or set judge.enabled: false, or pass --allow-uncalibrated)."
        )
        raise typer.Exit(2) from exc
    except LLMError as exc:
        typer.secho(f"LLM error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

    if allow_uncalibrated and metric_label.startswith("judge"):
        console.print(
            "[bold yellow]WARNING:[/bold yellow] --allow-uncalibrated — "
            "optimizing against a judge with NO valid calibration "
            "certificate. Scores may not track human judgement."
        )

    console.print(
        f"\nRunning [bold]{optimizer.value}[/bold] optimizer "
        f"on {len(examples_preview)} examples …"
    )
    console.print(f"[bold]Metric:[/bold] {metric_label}")
    console.print(f"[bold]Run id:[/bold] {run_id}")

    try:
        result = asyncio.run(coro)
    except LLMError as exc:
        typer.secho(f"LLM error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

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

    # Registration is handled inside _build_run_coro; just report the id.
    console.print(f"[bold]Registered prompt:[/bold] {result.best.id}")


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

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
    label_min: float | None = typer.Option(
        None,
        "--label-min",
        help="Declared minimum human-label value for binning (overrides observed min).",
    ),
    label_max: float | None = typer.Option(
        None,
        "--label-max",
        help="Declared maximum human-label value for binning (overrides observed max).",
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
    try:
        client = _build_client(cfg)
    except LLMError as exc:
        typer.secho(f"LLM error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    declared_range: tuple[float, float] | None = None
    if label_min is not None and label_max is not None:
        declared_range = (label_min, label_max)
    calibrator = Calibrator(
        judge, dataset, client, threshold_kappa=threshold, label_range=declared_range
    )

    console.print(
        f"\nCalibrating judge [bold]{judge_model}[/bold] on criterion "
        f"[bold]{criterion}[/bold] ({len(dataset)} gold records, "
        f"{len(calibrator.holdout)} holdout) …"
    )
    try:
        cert = asyncio.run(calibrator.calibrate())
    except LLMError as exc:
        typer.secho(f"LLM error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

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


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def _load_config_or_exit(config: str) -> PromptlineConfig:
    cfg_path = Path(config)
    if not cfg_path.exists():
        console.print(f"[red]Config not found:[/red] {cfg_path}")
        raise typer.Exit(2)
    return load_config(cfg_path)


def _load_split_or_exit(path: str, name: str) -> list[Example]:
    if not Path(path).exists():
        console.print(f"[red]{name} set not found:[/red] {path}")
        raise typer.Exit(2)
    return load_examples_jsonl(path)


@app.command()
def gate(
    candidate: list[str] = typer.Option(
        ...,
        "--candidate",
        help="Registered prompt id to challenge the incumbent (repeatable).",
    ),
    dev: str = typer.Option(..., "--dev", help="Dev split JSONL path."),
    val: str = typer.Option(..., "--val", help="Held-out val split JSONL path."),
    config: str = typer.Option(
        "promptline.yaml", "--config", help="Path to promptline.yaml."
    ),
) -> None:
    """Statistically gate candidates against the active prompt.

    Scores with the calibrated LLM-judge metric when ``judge.enabled`` (the
    default; requires a passing calibration certificate), otherwise with the
    exact-match metric.  Exit codes: 0 promote, 1 reject, 2 refusal.
    """
    from promptline.registry.gate import GateSettings, run_gate

    cfg = _load_config_or_exit(config)
    registry = PromptRegistry(Path(cfg.registry.path))
    program_name = cfg.program.name

    # ---- Incumbent (active prompt) -------------------------------------------
    active = registry.get_active(program_name)
    if active is None:
        console.print(
            "[red]No active prompt to gate against.[/red] "
            "Activate a baseline first: promptline registry activate <prompt_id>"
        )
        raise typer.Exit(2)
    incumbent_id, incumbent = active

    # ---- Candidates -----------------------------------------------------------
    candidates: list[Candidate] = []
    for cid in candidate:
        cand = registry.get(cid)
        if cand is None:
            console.print(f"[red]Unknown candidate prompt id:[/red] {cid}")
            raise typer.Exit(2)
        candidates.append(cand)

    # ---- Data -------------------------------------------------------------------
    dev_examples = _load_split_or_exit(dev, "dev")
    val_examples = _load_split_or_exit(val, "val")

    # ---- Gate ----------------------------------------------------------------------
    program, _ = _build_program_and_seed(cfg)
    try:
        client = _build_client(cfg)
    except LLMError as exc:
        typer.secho(f"LLM error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    harness = EvalHarness(client=client, cfg=_model_config(cfg))
    metric, metric_mode = resolve_metric(cfg, client)
    settings = GateSettings.from_config(cfg.gate, cfg.judge)
    if metric_mode == "judge" and settings.require_certificate_path is None:
        # Judge metric always requires a certificate: fall back to the
        # default location written by `promptline calibrate`.
        settings.require_certificate_path = resolve_certificate_path(cfg)
        settings.min_kappa = cfg.judge.min_kappa

    # Gate runs are I/O-bounded by data size, not by optimizer rollout count.
    # The rollout cap is deliberately None; only the cost ceiling is honored.
    gate_budget = Budget(max_rollouts=None, max_cost_usd=cfg.budget.max_cost_usd)

    console.print(
        f"\nGating {len(candidates)} candidate(s) against incumbent "
        f"[bold]{incumbent_id[:12]}[/bold] "
        f"({len(dev_examples)} dev / {len(val_examples)} val examples) …"
    )
    console.print(f"[bold]Metric:[/bold] {_metric_label(cfg, metric_mode)}")
    try:
        report = asyncio.run(
            run_gate(
                program=program,
                incumbent=incumbent,
                candidates=candidates,
                dev=dev_examples,
                val=val_examples,
                harness=harness,
                metric=metric,
                settings=settings,
                budget=gate_budget,
            )
        )
    except (ValueError, UncalibratedJudgeError) as exc:
        console.print(f"[red]Gate refused to run:[/red] {exc}")
        raise typer.Exit(2) from exc
    except LLMError as exc:
        typer.secho(f"LLM error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

    # ---- Report table ------------------------------------------------------------
    table = Table(title="Gate Report", show_lines=True)
    table.add_column("Candidate", style="dim", max_width=14)
    table.add_column("Δ mean", justify="right")
    table.add_column("CI", justify="right")
    table.add_column("p", justify="right")
    table.add_column("Significant", justify="center")
    for res in report.results:
        table.add_row(
            res.candidate_id[:12],
            f"{res.mean_delta:+.3f}",
            f"[{res.ci_low:+.3f}, {res.ci_high:+.3f}]",
            f"{res.p_value:.4f}",
            "[green]yes[/green]" if res.holm_significant else "[red]no[/red]",
        )
    console.print(table)

    verdict_lines = [f"Verdict: {report.verdict}"]
    if report.winner_id:
        verdict_lines.append(f"Winner: {report.winner_id}")
    if report.val_mean_delta is not None:
        verdict_lines.append(
            f"Val Δ: {report.val_mean_delta:+.3f} "
            f"[{report.val_ci_low:+.3f}, {report.val_ci_high:+.3f}]"
        )
    for flag in report.flags:
        verdict_lines.append(f"Flag: {flag}")
    for warning in report.warnings:
        verdict_lines.append(f"Warning: {warning}")
    console.print(
        Panel(
            "\n".join(verdict_lines),
            title="Gate Verdict",
            border_style="green" if report.verdict == "promote" else "red",
        )
    )

    # ---- Persist report --------------------------------------------------------------
    reports_dir = Path(cfg.registry.path) / "gate_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    report_path = reports_dir / f"gate-{stamp}-{uuid.uuid4().hex[:8]}.json"
    report_path.write_text(report.model_dump_json(indent=2))
    console.print(f"Report saved to [bold]{report_path}[/bold]")

    # ---- Promote ------------------------------------------------------------------------
    if report.verdict == "promote" and report.winner_id:
        registry.activate(
            program_name, report.winner_id, report.model_dump_json()
        )
        console.print(
            f"[green]Promoted[/green] {report.winner_id} to active for "
            f"program {program_name!r}"
        )
        raise typer.Exit(0)
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# registry sub-app
# ---------------------------------------------------------------------------

registry_app = typer.Typer(name="registry", help="Inspect and manage the prompt registry.")
app.add_typer(registry_app, name="registry")


def _open_registry(config: str) -> tuple[PromptRegistry, str]:
    cfg = _load_config_or_exit(config)
    return PromptRegistry(Path(cfg.registry.path)), cfg.program.name


@registry_app.command("list")
def registry_list_cmd(
    config: str = typer.Option("promptline.yaml", "--config"),
) -> None:
    """List registered prompts for the configured program."""
    registry, program_name = _open_registry(config)
    active = registry.get_active(program_name)
    active_id = active[0] if active else None
    table = Table(title=f"Prompts — program {program_name!r}")
    table.add_column("ID", style="dim")
    table.add_column("Created")
    table.add_column("Run")
    table.add_column("Score", justify="right")
    table.add_column("Active", justify="center")
    for row in registry.list_prompts(program_name):
        score = row["mean_score"]
        table.add_row(
            row["id"],
            row["created_at"][:19],
            row["run_id"][:12],
            f"{score:.3f}" if isinstance(score, float) else "—",
            "[green]●[/green]" if row["id"] == active_id else "",
        )
    console.print(table)


@registry_app.command("show")
def registry_show_cmd(
    prompt_id: str = typer.Argument(..., help="Registered prompt id."),
    config: str = typer.Option("promptline.yaml", "--config"),
) -> None:
    """Show a prompt's module instructions and lineage."""
    registry, _ = _open_registry(config)
    cand = registry.get(prompt_id)
    if cand is None:
        console.print(f"[red]Unknown prompt id:[/red] {prompt_id}")
        raise typer.Exit(1)
    for name, state in cand.modules.items():
        console.print(
            Panel(
                state.instruction,
                title=f"module {name!r} ({len(state.demos)} demos)",
            )
        )
    ancestors = registry.lineage(prompt_id)
    lineage_str = " -> ".join([prompt_id, *ancestors]) if ancestors else prompt_id
    console.print(f"[bold]Lineage:[/bold] {lineage_str}")


@registry_app.command("activate")
def registry_activate_cmd(
    prompt_id: str = typer.Argument(..., help="Registered prompt id."),
    config: str = typer.Option("promptline.yaml", "--config"),
) -> None:
    """Activate a prompt WITHOUT a gate report (baseline bootstrap)."""
    registry, program_name = _open_registry(config)
    try:
        registry.activate(program_name, prompt_id)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        "[yellow]Warning:[/yellow] activated without a gate report "
        "(baseline bootstrap). Use 'promptline gate' for gated promotions."
    )
    console.print(
        f"[green]Activated[/green] {prompt_id} for program {program_name!r}"
    )


@registry_app.command("rollback")
def registry_rollback_cmd(
    config: str = typer.Option("promptline.yaml", "--config"),
) -> None:
    """Revert the active pointer to the previous distinct prompt."""
    registry, program_name = _open_registry(config)
    try:
        target = registry.rollback(program_name)
    except RuntimeError as exc:
        console.print(f"[red]Rollback failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Rolled back[/green] program {program_name!r} to {target}"
    )


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def _web_dist_path() -> Path:
    """Location of the built dashboard (repo checkout layout)."""
    return Path(__file__).resolve().parents[2] / "web" / "dist"


def _warn_if_dashboard_missing() -> bool:
    """Print an amber warning when the dashboard is not built.

    Returns True when the warning was printed (dashboard missing).
    """
    if (_web_dist_path() / "index.html").exists():
        return False
    console.print(
        "[yellow]dashboard not built — run: "
        "cd web && npm install && npm run build "
        "(API still available)[/yellow]"
    )
    return True


def build_app_from_config(config_path: str):
    """Build the FastAPI app with real run_starter/gate_runner closures.

    Used by ``promptline serve`` and by tests (via TestClient, without
    spawning uvicorn).
    """
    from promptline.server.app import create_app
    from promptline.server.runs import RunManager

    cfg = load_config(config_path)
    registry = PromptRegistry(Path(cfg.registry.path))
    run_manager = RunManager(Path(cfg.registry.path) / "runs")

    def run_starter(spec, emit, run_dir: Path):
        """Build and return the run coroutine for a server-started optimizer pass.

        Receives ``(emit, run_dir)`` from :class:`~promptline.server.runs.RunManager`
        via the :data:`~promptline.server.runs.CoroFactory` protocol.

        Synchronous work (loading examples, constructing the LLM client) is
        performed here; errors surface immediately so ``POST /runs`` returns
        HTTP 400 (via :exc:`~promptline.server.runs.RunStartError`) rather than
        leaving a zombie run stuck at ``status='running'``.

        GEPA checkpointing is wired up via *run_dir*.  The ``run_id`` is
        ``run_dir.name`` (the leaf directory component assigned by RunManager).
        """
        choice = OptimizerChoice(spec.optimizer)
        data_path = spec.data_path or cfg.dataset.path
        run_budget = Budget(
            max_rollouts=(
                spec.budget if spec.budget is not None else cfg.budget.max_rollouts
            ),
            max_cost_usd=cfg.budget.max_cost_usd,
        )
        run_id = run_dir.name  # RunManager sets run_dir = base_dir / run_id
        coro, metric_label = _build_run_coro(
            cfg, choice, data_path, run_budget, emit, run_dir, run_id, registry
        )
        console.print(f"[bold]Metric:[/bold] {metric_label} (run {run_id})")
        return coro

    def gate_runner(payload: dict):
        from promptline.registry.gate import GateSettings, run_gate

        async def _run():
            program_name = payload.get("program") or cfg.program.name
            incumbent_id = payload.get("incumbent_id") or ""
            if incumbent_id:
                incumbent = registry.get(incumbent_id)
            else:
                active = registry.get_active(program_name)
                incumbent = active[1] if active else None
            if incumbent is None:
                raise ValueError("no incumbent prompt; activate a baseline first")
            candidates = []
            for cid in payload.get("candidate_ids", []):
                cand = registry.get(cid)
                if cand is None:
                    raise ValueError(f"unknown candidate prompt id: {cid}")
                candidates.append(cand)
            dev_examples = load_examples_jsonl(payload["dev_path"])
            val_examples = load_examples_jsonl(payload["val_path"])
            program, _ = _build_program_and_seed(cfg)
            client = _build_client(cfg)
            harness = EvalHarness(client=client, cfg=_model_config(cfg))
            metric, metric_mode = resolve_metric(cfg, client)
            settings = GateSettings.from_config(cfg.gate, cfg.judge)
            if metric_mode == "judge" and settings.require_certificate_path is None:
                settings.require_certificate_path = resolve_certificate_path(cfg)
                settings.min_kappa = cfg.judge.min_kappa
            # Parity with the CLI gate command: cost-capped budget, no
            # rollout ceiling (gate runs are bounded by data size).
            gate_budget = Budget(
                max_rollouts=None, max_cost_usd=cfg.budget.max_cost_usd
            )
            report = await run_gate(
                program=program,
                incumbent=incumbent,
                candidates=candidates,
                dev=dev_examples,
                val=val_examples,
                harness=harness,
                metric=metric,
                settings=settings,
                budget=gate_budget,
            )
            # Parity with the CLI: on a promote verdict, activate the winner
            # with the gate report attached (unless promote=false).
            activated = False
            if (
                payload.get("promote", True)
                and report.verdict == "promote"
                and report.winner_id
            ):
                registry.activate(
                    program_name, report.winner_id, report.model_dump_json()
                )
                activated = True
            return {**report.model_dump(), "activated": activated}

        return _run()

    # Serve the built dashboard when web/dist exists next to the repo root.
    web_dist = _web_dist_path()
    return create_app(
        registry,
        run_manager,
        run_starter=run_starter,
        gate_runner=gate_runner,
        web_dist=web_dist,
    )


@app.command()
def tui(
    run: str | None = typer.Option(
        None, "--run", help="Run id to watch (reads <registry>/runs/<id>/events.jsonl)."
    ),
    attach: str | None = typer.Option(
        None, "--attach", help="SSE URL to attach to (e.g. http://host/runs/<id>/events)."
    ),
    config: str = typer.Option(
        "promptline.yaml", "--config", help="Path to promptline.yaml."
    ),
) -> None:
    """Open the live TUI cockpit for an optimizer run."""
    from promptline.tui.app import PromptlineTUI
    from promptline.tui.events import RunEventFeed

    if attach:
        feed = RunEventFeed.from_url(attach)
        run_id = attach.rstrip("/").split("/")[-2] if "/" in attach else attach
    elif run:
        cfg = _load_config_or_exit(config)
        events_path = Path(cfg.registry.path) / "runs" / run / "events.jsonl"
        feed = RunEventFeed.from_file(events_path, follow=True)
        run_id = run
    else:
        console.print("[red]Pass --run <run_id> or --attach <url>.[/red]")
        raise typer.Exit(1)

    PromptlineTUI(feed=feed, run_id=run_id).run()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    config: str = typer.Option(
        "promptline.yaml", "--config", help="Path to promptline.yaml."
    ),
) -> None:
    """Serve the Promptline control and serving planes over HTTP."""
    if not Path(config).exists():
        console.print(f"[red]Config not found:[/red] {config}")
        raise typer.Exit(1)
    _warn_if_dashboard_missing()
    import uvicorn

    uvicorn.run(build_app_from_config(config), host=host, port=port)
