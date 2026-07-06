"""`promptline demo` — set up the support-assistant demo workspace.

Builds the datasets and config for the end-to-end demo story
(calibrate → optimize → gate → serve).  Online mode pulls HelpSteer2 and the
Bitext customer-support dataset from HuggingFace (requires the
``promptline[data]`` extra); ``--offline`` uses the bundled fixtures under
``examples/support-assistant/fixtures/`` instead.

File formats written into the workspace:

- ``gold.jsonl`` — judge gold set, Record schema with numeric ``human_label``.
- ``dev.jsonl`` / ``val.jsonl`` / ``feedback.jsonl`` — task splits in Record
  schema, with ``inputs``/``labels`` keys embedded on each line so the same
  files are directly consumable by ``promptline gate --dev/--val`` (which
  reads the optimize-format keys and ignores the Record keys).
- ``train.jsonl`` — optimize-format lines
  ``{"inputs": {"conversation": ...}, "labels": {"reference": ...}}`` for
  ``promptline optimize --data``.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from promptline.data.dataset import Dataset, Record

console = Console()

demo_app = typer.Typer(name="demo", help="Support-assistant demo pipeline.")

#: Bundled offline fixtures (repo checkout layout: <root>/examples/...).
FIXTURES_DIR = Path(__file__).resolve().parents[2] / "examples" / "support-assistant" / "fixtures"

#: Deliberately mediocre seed instruction — the optimizer's starting point.
SEED_INSTRUCTION = "You are a support agent. Answer the question."

#: Extra rows loaded beyond dev+val online; they land in the feedback split.
_FEEDBACK_SLACK = 200


def _record_to_optimize_row(record: Record) -> dict:
    """Optimize-format row: inputs.conversation / labels.reference."""
    transcript = "\n".join(f"{t.role}: {t.content}" for t in record.conversation)
    return {
        "inputs": {"conversation": transcript},
        "labels": {"reference": record.reference_output or ""},
        "meta": dict(record.meta),
    }


def _write_hybrid_jsonl(dataset: Dataset, path: Path) -> None:
    """Record schema + embedded optimize-format keys, one line per record.

    ``Dataset.from_jsonl`` validates the Record fields (extras ignored), while
    ``load_examples_jsonl`` reads the ``inputs``/``labels`` keys — so dev/val
    files work for both contamination checks and ``promptline gate``.
    """
    with path.open("w") as fh:
        for record in dataset:
            row = json.loads(record.model_dump_json())
            row.update(_record_to_optimize_row(record))
            fh.write(json.dumps(row) + "\n")


def _write_optimize_jsonl(dataset: Dataset, path: Path) -> None:
    with path.open("w") as fh:
        for record in dataset:
            fh.write(json.dumps(_record_to_optimize_row(record)) + "\n")


def _demo_config_yaml() -> str:
    """The demo promptline.yaml (paths relative to the workspace dir)."""
    return """\
# promptline.yaml — support-assistant demo
# Run promptline commands from inside this directory.

program:
  name: support
  # Deliberately mediocre seed prompt: the optimizer's job is to beat it.
  instruction: "You are a support agent. Answer the question."
  inputs:
    - conversation
  outputs:
    - answer

models:
  task: meta-llama/llama-3.1-8b-instruct
  reflection: anthropic/claude-3.5-haiku
  judge: anthropic/claude-3.5-haiku

dataset:
  kind: jsonl
  path: train.jsonl

budget:
  max_rollouts: 300
  max_cost_usd: 5.0

judge:
  # The calibrated helpfulness judge is the optimize/gate metric.
  # `promptline calibrate` writes the certificate exactly where the
  # optimizer expects it — calibration genuinely unlocks optimization.
  enabled: true
  criterion: helpfulness
  certificate: .promptline/certificates/helpfulness.json
  min_kappa: 0.6

gate:
  alpha: 0.05
  min_examples: 50

registry:
  path: .promptline
"""


def _load_online(gold_n: int, support_n: int) -> tuple[Dataset, Dataset]:
    try:
        from promptline.data.loaders import load_bitext, load_helpsteer2

        gold = load_helpsteer2(limit=gold_n)
        support = load_bitext(limit=support_n)
    except ImportError as exc:
        console.print(
            f"[red]{exc}[/red]\n"
            "Install the data extra ([bold]pip install 'promptline[data]'[/bold]) "
            "or rerun with [bold]--offline[/bold] to use bundled fixtures."
        )
        raise typer.Exit(1) from exc
    return gold, support


def _load_offline(gold_n: int) -> tuple[Dataset, Dataset]:
    gold_path = FIXTURES_DIR / "gold_fixture.jsonl"
    support_path = FIXTURES_DIR / "support_fixture.jsonl"
    if not gold_path.exists() or not support_path.exists():
        console.print(f"[red]Bundled fixtures not found under[/red] {FIXTURES_DIR}")
        raise typer.Exit(1)
    gold = Dataset.from_jsonl(gold_path)
    gold = Dataset(gold.records[:gold_n])
    support = Dataset.from_jsonl(support_path)
    return gold, support


@demo_app.command("setup")
def demo_setup(
    dir: str = typer.Option(
        "examples/support-assistant/workspace",
        "--dir",
        help="Workspace directory to create.",
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Use bundled fixtures instead of HuggingFace."
    ),
    gold_n: int = typer.Option(
        400, "--gold-n", help="Max HelpSteer2 gold records for judge calibration."
    ),
    dev_n: int = typer.Option(150, "--dev-n", help="Target size of the dev split."),
    val_n: int = typer.Option(150, "--val-n", help="Target size of the val split."),
) -> None:
    """Build the support-assistant demo workspace: datasets + promptline.yaml."""
    if gold_n <= 0 or dev_n <= 0 or val_n <= 0:
        console.print("[red]--gold-n, --dev-n and --val-n must be positive.[/red]")
        raise typer.Exit(1)

    workspace = Path(dir)
    workspace.mkdir(parents=True, exist_ok=True)

    # ---- Load source data ---------------------------------------------------
    support_n = dev_n + val_n + _FEEDBACK_SLACK
    if offline:
        gold, support = _load_offline(gold_n)
    else:
        gold, support = _load_online(gold_n, support_n)

    # ---- Gold set (judge calibration) ----------------------------------------
    gold.to_jsonl(workspace / "gold.jsonl")

    # ---- Seeded split: dev / val / feedback -----------------------------------
    dev_frac = dev_n / support_n
    val_frac = val_n / support_n
    splits = support.split(
        {"dev": dev_frac, "val": val_frac, "feedback": 1.0 - dev_frac - val_frac},
        seed=0,
    )
    _write_hybrid_jsonl(splits["dev"], workspace / "dev.jsonl")
    _write_hybrid_jsonl(splits["val"], workspace / "val.jsonl")
    _write_hybrid_jsonl(splits["feedback"], workspace / "feedback.jsonl")

    # ---- Optimizer training data (optimize-format) ----------------------------
    _write_optimize_jsonl(splits["feedback"], workspace / "train.jsonl")

    # ---- Config ----------------------------------------------------------------
    (workspace / "promptline.yaml").write_text(_demo_config_yaml())

    # ---- Summary + next steps -----------------------------------------------------
    console.print(
        f"\n[green]Demo workspace ready:[/green] {workspace.resolve()}\n"
        f"  gold.jsonl      {len(gold)} records (judge calibration)\n"
        f"  dev.jsonl       {len(splits['dev'])} records\n"
        f"  val.jsonl       {len(splits['val'])} records\n"
        f"  feedback.jsonl  {len(splits['feedback'])} records\n"
        f"  train.jsonl     {len(splits['feedback'])} examples (optimize format)"
    )
    console.print(
        Panel(
            f"cd {dir}\n"
            "export OPENROUTER_API_KEY=sk-or-...\n"
            "\n"
            "# 1. Calibrate the helpfulness judge against gold human labels\n"
            "#    (writes the certificate that unlocks optimize/gate)\n"
            "promptline calibrate --gold gold.jsonl --label-min 0 --label-max 4\n"
            "\n"
            "# 2. Optimize the seed support prompt with GEPA "
            "(judge-scored, certificate-gated)\n"
            "promptline optimize --optimizer gepa --data train.jsonl\n"
            "\n"
            "# 3. Gate the winner against the active baseline\n"
            "promptline registry list\n"
            "promptline registry activate <seed_prompt_id>   # baseline bootstrap\n"
            "promptline gate --candidate <best_prompt_id> "
            "--dev dev.jsonl --val val.jsonl\n"
            "\n"
            "# 4. Serve the promoted prompt\n"
            "promptline serve\n"
            "curl http://127.0.0.1:8000/prompts/support/active",
            title="NEXT STEPS",
            border_style="cyan",
        )
    )
