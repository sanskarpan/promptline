# Promptline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build Promptline — calibrated LLM-judge → from-scratch prompt optimizers (GEPA flagship) → statistical deploy gate → versioned registry + serving — per `docs/superpowers/specs/2026-07-03-promptline-design.md`.

**Architecture:** Library-core with thin shells. One Python package `promptline` (core, data, judge, optimizers, eval, registry, server, tui, cli); React dashboard in `web/` served static by FastAPI. All LLM traffic through `LLMClient` (OpenRouter adapter, BYO key). Every optimizer implements one contract and emits typed events consumed by TUI/dashboard over SSE.

**Tech Stack:** Python 3.11+, uv, pytest, Ruff, pyright, httpx, pydantic v2, Typer, Textual, FastAPI + sse-starlette, Optuna (TPE), numpy/scipy, datasets (HF), React + Vite + TypeScript.

**Conventions for every task (do not repeat in tasks):**
- TDD: write the failing test first, run it (expect FAIL), implement minimally, run again (expect PASS), then `git add <files> && git commit`.
- Test command: `uv run pytest tests/<path> -v`. Lint before commit: `uv run ruff check . && uv run ruff format .`.
- All public dataclasses/models are pydantic v2 or `@dataclass(frozen=True)` where noted. Full type hints everywhere.
- No LLM network calls in unit tests — use `FakeLLMClient`.
- Commit messages: conventional commits (`feat:`, `test:`, `chore:`).

---

## Phase 1 — Core, eval harness, BootstrapFewShot, OPRO, CLI skeleton

### Task 1: Project scaffolding

**Files:** Create `pyproject.toml`, `promptline/__init__.py`, `tests/__init__.py`, `.gitignore`, `README.md` (stub), `ruff.toml`.

- [x] `uv init --package --name promptline`, set `requires-python = ">=3.11"`.
- [x] Add deps: `httpx pydantic typer rich textual fastapi sse-starlette uvicorn optuna numpy scipy` and dev deps `pytest pytest-asyncio ruff pyright respx`. HF `datasets` goes in an optional extra `[data]`.
- [x] Create package subdirs with `__init__.py`: `core data judge optimizers eval registry server tui cli`.
- [x] Smoke test `tests/test_import.py`: `import promptline` and every subpackage; assert `promptline.__version__ == "0.1.0"`.
- [x] Commit `chore: scaffold promptline package`.

### Task 2: Core types (`promptline/core/types.py`, test `tests/core/test_types.py`)

- [x] Define:
  - `Field(name: str, desc: str = "")`
  - `Signature(instruction: str, inputs: list[Field], outputs: list[Field])` — `.render_system()` produces the system prompt: instruction + I/O field spec; `.parse_output(text) -> dict[str, str]` parses `[[field]]: value` sections (one repair path: if exactly one output field, whole text is its value).
  - `Example(inputs: dict[str, str], labels: dict[str, str] = {}, meta: dict = {})`
  - `Demo(inputs: dict[str, str], outputs: dict[str, str])`
  - `ModuleState(instruction: str, demos: list[Demo])`
  - `Candidate(id: str, modules: dict[str, ModuleState], parent_ids: list[str], optimizer: str, meta: dict)` — `Candidate.child(**changes)` helper creates a new candidate with fresh uuid and `parent_ids=[self.id]`.
- [x] Tests: render_system includes instruction and field names; parse_output round-trips two fields; parse falls back for single output field; `child()` sets lineage.
- [x] Commit `feat: core signature/candidate types`.

### Task 3: LLMClient interface + FakeLLMClient (`core/llm.py`, test `tests/core/test_llm.py`)

- [x] `LLMCall(model, messages, temperature, max_tokens, seed?)` (frozen, hashable via `.key()` = sha256 of canonical JSON). `LLMResponse(text, prompt_tokens, completion_tokens, cost_usd, cached: bool)`.
- [x] `class LLMClient(Protocol): async def complete(self, call: LLMCall) -> LLMResponse`.
- [x] `FakeLLMClient(script: list[str] | Callable[[LLMCall], str])` — returns scripted responses in order (or via callable), records `self.calls: list[LLMCall]`, zero cost, deterministic.
- [x] Tests: scripted order, call recording, callable mode, `.key()` stable across dict ordering.
- [x] Commit `feat: LLM client protocol and fake client`.

### Task 4: SQLite call cache (`core/cache.py`, test `tests/core/test_cache.py`)

- [x] `LLMCache(path)` — table `calls(key TEXT PRIMARY KEY, response_json TEXT, created_at)`. `get(call) -> LLMResponse | None` (sets `cached=True`), `put(call, resp)`. `CachingClient(inner: LLMClient, cache: LLMCache)` wraps any client; cache hit skips inner.
- [x] Tests: miss→put→hit; hit does not call inner (use FakeLLMClient and count calls); persists across reopen (tmp_path).
- [x] Commit `feat: sqlite llm call cache`.

### Task 5: OpenRouter adapter (`core/openrouter.py`, test `tests/core/test_openrouter.py` using `respx`)

- [x] `OpenRouterClient(api_key, base_url="https://openrouter.ai/api/v1", max_retries=4)` implements `LLMClient` via httpx `POST /chat/completions` with `usage: {include: true}`; reads `usage.cost` for `cost_usd` (fallback 0.0). Exponential backoff with jitter on 429/5xx/timeouts; raise `LLMError` after retries. Env fallback `OPENROUTER_API_KEY`.
- [x] Tests (respx): happy path maps text+usage+cost; 429 then 200 retries; 500×5 raises `LLMError`; auth header set.
- [x] Commit `feat: openrouter adapter with retries and cost tracking`.

### Task 6: PromptProgram + tracing (`core/program.py`, test `tests/core/test_program.py`)

- [x] `Module(name, signature)`; `PromptProgram(modules: dict[str, Module])` with `simple(name="main", instruction, inputs, outputs)` classmethod for the 1-module case.
- [x] `await program.run(example, candidate, client, model_cfg) -> Prediction(outputs, traces, cost_usd)`. Execution per module: system = candidate.modules[name] instruction rendered + demos as user/assistant turn pairs; user = example inputs rendered. Malformed structured output → one repair reprompt ("Your output was not in the required format...") → else `Prediction.failed(reason)`.
- [x] `Trace(module, system_prompt, user_prompt, raw_output, parsed: dict | None)` appended per module call. Multi-module programs run modules in insertion order, piping prior outputs into later inputs when field names match.
- [x] `ModelConfig(task_model, temperature=0.2, max_tokens=1024)`.
- [x] Tests with FakeLLMClient: single-module run parses outputs and records trace; demos rendered as turns; repair path invoked once then failure; two-module piping.
- [x] Commit `feat: prompt program execution with traces`.

### Task 7: Metric contract + eval harness (`eval/harness.py`, test `tests/eval/test_harness.py`)

- [x] `MetricResult(score: float, feedback: str = "", per_module: dict[str, str] = {})`; `Metric = Callable[[Example, Prediction], Awaitable[MetricResult]]` (sync allowed, wrapped).
- [x] `Budget(max_rollouts: int | None, max_cost_usd: float | None)` — `.charge(rollouts=1, cost=x)`, `.exhausted`, thread-safe; `BudgetExhausted` raised at the harness boundary only (never mid-example).
- [x] `EvalHarness(client, model_cfg, concurrency=8)`: `await harness.evaluate(program, candidate, examples, metric, budget) -> EvalReport(per_example: list[ExampleResult(example_idx, score, feedback, cost)], mean_score, total_cost)`. Failed predictions score 0.0 with feedback = failure reason. Every example charged to budget; stops launching new examples when exhausted, returns partial report flagged `truncated=True`.
- [x] Tests: mean over fake scores; budget stops early and flags truncated; failed prediction scores 0; concurrency respected (semaphore counter test).
- [x] Commit `feat: eval harness with hard budget`.

### Task 8: Statistics (`eval/stats.py`, test `tests/eval/test_stats.py`)

- [x] Implement exactly:
  ```python
  def paired_bootstrap_ci(deltas: np.ndarray, n_boot=10_000, alpha=0.05, rng_seed=0) -> tuple[float, float, float]:
      """Returns (mean_delta, ci_low, ci_high) via percentile bootstrap on the mean."""
  def holm_correct(pvals: list[float], alpha=0.05) -> list[bool]:
      """Holm-Bonferroni: sort ascending, reject while p_i <= alpha/(m-i)."""
  def bootstrap_pvalue(deltas: np.ndarray, n_boot=10_000, rng_seed=0) -> float:
      """Two-sided: shift deltas to mean 0, p = frac(|boot means| >= |observed mean|)."""
  def min_examples_warning(n: int, floor: int = 50) -> str | None
  ```
- [x] Tests: CI on N(0.1, 0.01, n=500) excludes 0; CI on N(0,1,n=30) includes 0; Holm known vector `[0.01, 0.04, 0.03]` with alpha .05 → `[True, False, False]` (verify by hand: sorted .01≤.0167 ✓, .03≤.025 ✗ stop); p-value calibration: for null deltas p>0.05 in ≥90% of 20 seeded reps.
- [x] Commit `feat: paired bootstrap and holm correction`.

### Task 9: Optimizer contract + run events (`optimizers/base.py`, test `tests/optimizers/test_base.py`)

- [x] `RunEvent(type: Literal["run_started","candidate_proposed","minibatch_scored","full_eval","pareto_updated","merge_attempted","budget_tick","run_finished"], payload: dict, ts: float)`.
- [x] `class Optimizer(ABC): name: str; async def optimize(self, program, seed_candidate, trainset, metric, budget, harness, emit: Callable[[RunEvent], None]) -> OptimizeResult(best: Candidate, candidates: list[Candidate], scores: dict[str, float], events_count: int)`.
- [x] `RunRecorder` — collects events + writes JSONL to run dir (`runs/<run_id>/events.jsonl`, `checkpoint.json`).
- [x] Tests: recorder writes/reads JSONL; abstract contract enforced.
- [x] Commit `feat: optimizer contract and run events`.

### Task 10: BootstrapFewShot + RandomSearch (`optimizers/bootstrap.py`, test `tests/optimizers/test_bootstrap.py`)

- [x] `BootstrapFewShot(max_demos=4, threshold=1.0)`: run seed candidate over shuffled trainset; each example whose `MetricResult.score >= threshold` yields a `Demo` from its trace (module inputs/outputs); stop at `max_demos` per module; return candidate with demos attached.
- [x] `BootstrapRandomSearch(n_subsets=8, subset_size=4, val_fraction=0.3)`: bootstrap a demo pool (up to `n_subsets*subset_size` passing traces), sample `n_subsets` random demo subsets (seeded rng), evaluate each on the val split, return best.
- [x] Tests with FakeLLMClient + scripted metric: only passing traces become demos; cap respected; random search picks the subset the scripted metric favors; budget charged.
- [x] Commit `feat: bootstrap fewshot optimizers`.

### Task 11: OPRO (`optimizers/opro.py`, test `tests/optimizers/test_opro.py`)

- [x] `OPRO(n_steps=10, candidates_per_step=4, minibatch_size=None)` — meta-prompt = task description + top-20 (instruction, score) trajectory sorted ascending + "write a new instruction better than all above, in <INS></INS> tags". Proposer uses `model_cfg.reflection_model`. Evaluate each proposal (full trainset or minibatch), keep trajectory, return best. Documented caveat in docstring: needs a strong proposer model.
- [x] Tests: trajectory grows and stays sorted; `<INS>` parsing (with fallback: whole reply); best returned; events emitted (`candidate_proposed`, `minibatch_scored`).
- [x] Commit `feat: OPRO baseline optimizer`.

### Task 12: Config + CLI skeleton (`core/config.py`, `cli/main.py`, test `tests/cli/test_cli.py`)

- [x] `PromptlineConfig` (pydantic, loaded from `promptline.yaml`): `program{name, instruction, inputs, outputs}`, `models{task, reflection, judge}`, `dataset{kind, path?}`, `budget{max_rollouts, max_cost_usd}`, `gate{alpha=0.05, min_examples=50}`, `registry{path=".promptline"}`.
- [x] Typer app `promptline`: `init` (writes commented sample yaml), `optimize --optimizer [bootstrap|bootstrap-rs|opro] --budget N` (wires config → harness → optimizer → prints best instruction + score table via rich), `version`.
- [x] Tests (CliRunner): `init` writes parseable config; `optimize` end-to-end with FakeLLMClient injected via a `--fake` hidden flag (env `PROMPTLINE_FAKE_SCRIPT` pointing at a JSON script file) completes and prints a score.
- [x] Commit `feat: config and cli skeleton`.

## Phase 2 — Data, judge, calibration

### Task 13: Dataset schema + JSONL adapter (`data/dataset.py`, test `tests/data/test_dataset.py`)

- [x] `Record(conversation: list[{role, content}], reference_output: str | None, human_label: float | dict | None, meta)`; `Dataset(records).to_examples(input_key="conversation") -> list[Example]`; `Dataset.from_jsonl(path)` / `.to_jsonl`; deterministic `split(fractions, seed) -> dict[str, Dataset]` with hash-based assignment (stable across runs); `contamination_check(a, b)` on content hashes.
- [x] Tests: jsonl round-trip; split stable + disjoint; contamination detected.
- [x] Commit `feat: dataset schema and jsonl adapter`.

### Task 14: HF loaders (`data/loaders.py`, test `tests/data/test_loaders.py` — unit tests use fixture dicts, not network)

- [x] `load_helpsteer2(attribute="helpfulness") -> Dataset` — maps rows to Records: conversation=[user prompt], reference_output=response, human_label=rating 0–4. `load_bitext() -> Dataset` — conversation=[instruction], reference_output=response, meta.intent. `load_mtbench_human() -> Dataset` — pairwise: meta carries both answers + human winner. Each guarded by `try: import datasets` with actionable error (`pip install promptline[data]`).
- [x] Row-mapper functions are pure (`_map_helpsteer_row(row) -> Record` etc.) and unit-tested on literal sample rows copied from the HF dataset viewer.
- [x] Commit `feat: helpsteer2/bitext/mtbench loaders`.

### Task 15: Judge (`judge/judge.py`, test `tests/judge/test_judge.py`)

- [x] `RubricCriterion(name, description, scale: tuple[int,int] = (1,5), anchors: dict[int,str] = {})`.
- [x] `Judge.pointwise(criterion, judge_model)` — a `PromptProgram` whose signature: inputs `conversation, response` (+`reference` optional), outputs `reasoning, score`; instruction template: role, rubric with anchors, "reason step by step, then output [[score]]: <int>", "do not reward length". `await judge.score(record, response, client) -> JudgeScore(value: float, reasoning)` (k-sample mean when `samples=k`, temperature 0 when k=1).
- [x] `Judge.pairwise(criterion, judge_model)` — inputs `conversation, response_a, response_b`, output verdict A/B/TIE; `score_pair()` runs both orderings, disagreement → TIE.
- [x] Tests with FakeLLMClient: score parsing incl. out-of-range clamp; k-sample mean; pairwise swap consistency → verdict, swap disagreement → TIE.
- [x] Commit `feat: pointwise and pairwise judges`.

### Task 16: Agreement metrics (`judge/metrics.py`, test `tests/judge/test_metrics.py`)

- [x] `cohens_kappa(a, b, weights=None|"quadratic")`, `spearman(a, b)` (scipy), `pairwise_accuracy(judge_verdicts, human_verdicts, ignore_ties=True)`.
- [x] Tests against hand-computed 2×2 kappa, scipy cross-check, known accuracy vector.
- [x] Commit `feat: judge agreement metrics`.

### Task 17: Calibrator + certificate (`judge/calibrator.py`, test `tests/judge/test_calibrator.py`)

- [x] `Calibrator(judge, gold: Dataset, client)`: splits gold → judge-dev/judge-holdout (default 0.5/0.5, seed fixed); `await calibrate() -> CalibrationCertificate(kappa, spearman, n_holdout, threshold, passed: bool, judge_candidate_id, created_at, per_score_confusion)`. Scalar labels are binned to the judge scale for kappa (document binning). Optional `meta_optimize(optimizer)` — optimizes the judge instruction on judge-dev with metric = agreement with human label, then re-certifies on holdout (holdout never used in optimization).
- [x] Certificate persisted as JSON in registry dir; `require_certificate(judge_id, min_kappa)` helper raises `UncalibratedJudgeError`.
- [x] Tests: perfect fake judge → kappa 1.0, passed; scripted disagreement → fails threshold; meta_optimize never touches holdout (assert via recorded calls); certificate round-trips JSON.
- [x] Commit `feat: judge calibrator with certificates`.

### Task 18: CLI calibrate + dataset commands

- [x] `promptline calibrate --gold helpsteer2 --criterion helpfulness` → runs Calibrator (subset size flag `--n 200`), prints certificate table; `promptline data prepare --demo` stub for now (full demo in Task 29).
- [x] CliRunner test with fake client script.
- [x] Commit `feat: calibrate cli`.

## Phase 3 — GEPA + ProTeGi

### Task 19: GEPA state + Pareto selection (`optimizers/gepa/state.py`, `pareto.py`, tests `tests/optimizers/gepa/test_pareto.py`)

- [x] `GepaState(candidates: dict[id, Candidate], scores: dict[id, list[float|None]]  # per pareto-instance, parents, lineage)`.
- [x] Implement Algorithm 2 exactly:
  ```python
  def pareto_sample(state, rng) -> str:
      # s*[i] = max over candidates of score[c][i]
      # P*[i] = {c : score[c][i] == s*[i]}
      # C = union of P*[i]; remove dominated (c dominated if some other c' in C
      #     has score >= on ALL instances and c never uniquely best)
      # f[c] = #instances where c in frontier after pruning
      # sample c with prob proportional to f[c]
  ```
- [x] Tests: single dominant candidate always chosen; specialist candidate (best on 1 instance only) has nonzero probability (seeded rng, frequency check over 200 draws); dominated candidate never sampled.
- [x] Commit `feat: gepa per-instance pareto selection`.

### Task 20: GEPA reflection + merge (`optimizers/gepa/reflect.py`, `merge.py`, tests)

- [x] `build_reflection_prompt(module_name, current_instruction, traces_with_feedback: list) -> str` — includes per-example: inputs, module output, score, textual feedback; asks for diagnosis then new instruction in a fenced code block. `parse_new_instruction(reply)` (fenced block, fallback whole reply).
- [x] Merge (Appendix F triplet rule):
  ```python
  def merge_candidates(ancestor, p1, p2, scores) -> dict[str, ModuleState]:
      # per module: if exactly one parent differs from ancestor -> that parent's
      # if both differ -> higher-scoring parent's (rng tie-break)
      # if neither differs -> p1's
  def common_ancestor(state, a, b) -> str | None  # walk parent pointers, BFS
  ```
- [x] Tests: triplet rule truth table (all 4 cases); common ancestor on a diamond lineage; reflection prompt contains traces and feedback; parse fallback.
- [x] Commit `feat: gepa reflection and system-aware merge`.

### Task 21: GEPA engine (`optimizers/gepa/engine.py`, test `tests/optimizers/gepa/test_engine.py`)

- [x] `GEPA(minibatch_size=3, n_pareto=32, use_merge=True, max_merges=5, accept="strict")` implementing Algorithm 1: split trainset → D_feedback/D_pareto; full-eval seed on D_pareto; loop until budget: pareto_sample → round-robin module → minibatch run w/ traces+feedback → reflect → new candidate → accept iff minibatch score improves (strict >) → full-eval accepted candidate on D_pareto; merge scheduled every `merge_every=7` accepted candidates when lineages differ. Emit all event types. Checkpoint state each acceptance; `GEPA.resume(run_dir)`.
- [x] Tests with scripted fake (metric rewards presence of a keyword the scripted reflection inserts): seed → improved candidate accepted → best returned; budget wall respected exactly (rollout count assertion); resume from checkpoint continues candidate ids; merge path exercised on forced lineage.
- [x] Commit `feat: gepa engine`.

### Task 22: ProTeGi + racing (`optimizers/protegi.py`, test `tests/optimizers/test_protegi.py`)

- [x] `ProTeGi(beam_width=4, n_gradients=2, n_paraphrases=1, racing_rounds=3, racing_batch=8)`: per beam candidate — collect failing examples on a minibatch → LLM critique ("textual gradient") → LLM edit applying critique → paraphrase expansion; candidate pool scored by successive-halving racing (each round evaluates survivors on a fresh batch, drop bottom half; final survivors full-evaled). Return best.
- [x] Tests: failing examples selected correctly; racing drops scripted losers and total rollouts < full-eval-everything (arithmetic assertion); beam bounded.
- [x] Commit `feat: protegi textual-gradient optimizer with racing`.

## Phase 4 — Registry + gate + serving

### Task 23: Registry (`registry/registry.py`, test `tests/registry/test_registry.py`)

- [x] SQLite at `<registry.path>/registry.db`: tables `prompts(id, program, candidate_json, created_at, run_id, parent_ids)`, `evals(prompt_id, dataset_hash, mean_score, n, report_json)`, `active(program, prompt_id, activated_at, gate_report_json)` + `history`. API: `register(candidate, program, run_id)`, `record_eval(...)`, `get_active(program)`, `activate(program, prompt_id, gate_report)` (only path that moves the pointer), `rollback(program)` (previous history entry), `lineage(prompt_id)`.
- [x] Property-ish tests: activate/rollback sequence invariants (rollback after N activations returns N-1th; rollback on empty history errors); lineage walk; eval records append-only.
- [x] Commit `feat: sqlite prompt registry`.

### Task 24: Gate (`registry/gate.py`, test `tests/registry/test_gate.py`)

- [x] `run_gate(incumbent, candidates: list, program, dataset_dev, dataset_val, harness, metric, cfg) -> GateReport`:
  1. refuse if judge metric lacks passing certificate (`UncalibratedJudgeError`), `len(dev) < cfg.min_examples`, or contamination between dev/val;
  2. evaluate incumbent + each candidate on dev (paired, same examples, cache makes incumbent cheap);
  3. per candidate: deltas → `bootstrap_pvalue`; Holm across candidates; survivors ranked by mean delta;
  4. winner confirmed on untouched val: `paired_bootstrap_ci` must exclude 0;
  5. verdict `promote|reject` + tripwires: mean output length ratio > 1.5 → `verbosity_flag`; attach 5 sampled outputs for human spot-check.
- [x] `GateReport` (pydantic): per-candidate deltas/CI/p, holm mask, winner, val CI, flags, verdict — JSON-serializable for registry + dashboard.
- [x] Tests (all with synthetic scripted scores): clear winner promotes; noise-only rejects; multiple-candidate Holm scenario (one true winner among 5 nulls — false-promotion rate check over seeds); undersized dev refuses; verbosity flag fires.
- [x] Commit `feat: statistical deploy gate`.

### Task 25: Serving + control API (`server/app.py`, `server/events.py`, test `tests/server/test_api.py`)

- [x] FastAPI `create_app(registry, run_manager)`:
  - Serving: `GET /prompts/{program}/active` → `{prompt_id, modules, activated_at}` with ETag (sha of prompt_id); 304 on match; 404 if none.
  - Control: `POST /runs` (start optimize in background task), `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/events` (SSE, replays JSONL then tails), `POST /gate`, `GET /registry/{program}`, `POST /registry/{program}/activate`, `POST /registry/{program}/rollback`, `GET /judges/certificates`.
  - `RunManager` — asyncio tasks keyed by run_id over the same optimizer entrypoints the CLI uses.
- [x] Tests (TestClient + fake client): active endpoint + ETag/304; run lifecycle with fake optimizer completes; SSE replays recorded events (read N events then disconnect); activate via API moves pointer.
- [x] Commit `feat: fastapi control and serving planes`.

### Task 26: CLI completion (`cli/main.py` additions, test extends `tests/cli/test_cli.py`)

- [x] Add `gate`, `registry list|show|activate|rollback`, `serve` (uvicorn), `optimize --optimizer gepa|protegi|mipro` wiring, `--resume`.
- [x] CliRunner tests for registry commands against tmp registry; gate command with scripted scores.
- [x] Commit `feat: full cli surface`.

## Phase 5 — MIPRO-like

### Task 27: MIPRO (`optimizers/mipro.py`, test `tests/optimizers/test_mipro.py`)

- [x] Stage 1: reuse `BootstrapFewShot` to build `n_candidates` demo sets per module.
- [x] Stage 2: grounded proposal — LLM writes dataset summary (from 10 sampled examples) + program summary (from signatures); proposer prompt = summaries + bootstrapped demos + history + random tip from `["creative","simple","descriptive","high_stakes","persona"]` (seeded); generate `n_candidates` instructions per module (original = candidate 0).
- [x] Stage 3: Optuna `TPESampler(seed=...)`, categorical per module × {instruction_idx, demoset_idx}; each trial evaluates a minibatch (default 16); every `full_eval_steps=5` trials, full-eval the best-mean config; return best fully-evaled candidate. `n_trials` from budget.
- [x] Tests: scripted setup where instruction #2 + demoset #1 is strictly best — TPE finds it within 30 trials; dataset summary prompt contains sampled examples; budget accounting includes trials + full evals.
- [x] Commit `feat: mipro-like bayesian optimizer`.

## Phase 6 — TUI + dashboard

### Task 28: TUI (`tui/app.py`, test `tests/tui/test_tui.py` via `textual.testing`)

- [x] Textual app `promptline tui [--run <id>|--attach <url>]`: panes — score sparkline + best-so-far, candidate lineage tree (Tree widget), live event log (RichLog with per-call cost), budget progress bar. Data source: run dir JSONL tail (local) or SSE (attach). Dark, monospace, flat borders (design language: opencode/Hermes — no rounded, single accent color).
- [x] Tests: app mounts; feeding 10 synthetic events updates tree node count and budget bar (Textual pilot).
- [x] Commit `feat: textual tui cockpit`.

### Task 29: Web dashboard (`web/`, e2e in Task 32)

- [x] Scaffold Vite + React + TS in `web/`; dark terminal theme tokens (bg #0a0a0a, panel #111, border #2a2a2a 1px solid, mono font stack `ui-monospace, 'JetBrains Mono', monospace`, accent single green `#4af6c3`-family, semantic red; sharp corners everywhere).
- [x] Pages (react-router): **Runs** (list + live run view: SSE score curve via lightweight SVG chart, cost/budget meters, event feed), **Lineage** (tree via `d3-hierarchy` or ELK-free custom layout; node click → prompt text + diff vs parent using `diff` npm pkg + per-example score strip), **Judge** (certificate card, scatter judge-vs-human, confusion matrix grid, disagreement table), **Gate** (delta histogram, CI bar, verdict banner, promote/rollback buttons → POST), **Registry** (version table, active badge).
- [x] API client from a typed `api.ts`; Vite dev proxy → 8000; `npm run build` outputs to `web/dist`; FastAPI mounts `StaticFiles(web/dist)` at `/` when present.
- [x] Component tests (vitest) for score-curve reducer and diff view; build wired into `promptline serve` docs.
- [x] Commit `feat: react dashboard with terminal design language`.

## Phase 7 — Demo, docs, benchmark

### Task 30: Demo pipeline (`examples/support-assistant/`, `cli demo`)

- [x] `promptline demo setup` — downloads HelpSteer2 + Bitext via loaders, builds splits (judge gold 400 / dev 150 / val 150 / feedback pool), writes `promptline.yaml` with cheap OpenRouter defaults (task: `meta-llama/llama-3.1-8b-instruct`, reflection+judge: `anthropic/claude-3.5-haiku` — different family), seed prompt = deliberately mediocre ("You are a support agent. Answer the question.").
- [x] `examples/support-assistant/README.md` — the 4-step walkthrough from the spec, with expected outputs.
- [x] Test: demo setup with `--offline` flag uses bundled 50-row fixture JSONLs (committed) instead of HF; full offline demo runs with FakeLLMClient script.
- [x] Commit `feat: support-assistant demo`.

### Task 31: Docs + README

- [x] README: hero = before/after prompt diff + gate report screenshot placeholder, quickstart, architecture diagram (the spec's ASCII), optimizer table, differentiation paragraph, links. `docs/`: concepts page per subsystem (core, judge, optimizers incl. algorithm notes + paper citations, gate math, API reference via mkdocs-style headings — plain markdown, no site generator yet).
- [x] Commit `docs: readme and concept docs`.

## Phase 8 — In-depth E2E testing

### Task 32: Offline E2E suite (`tests/e2e/`)

- [x] `test_full_pipeline_offline.py` — the entire chain with FakeLLMClient scripts: load fixture data → calibrate judge (scripted agreement → cert passes) → GEPA optimize (scripted reflection improves keyword metric) → gate (scripted winner) → registry activate → FastAPI TestClient fetches active prompt == winner. Asserts: cert κ, candidate lineage depth ≥ 2, gate verdict promote, ETag flow.
- [x] Same chain for `bootstrap-rs`, `protegi`, `mipro`, `opro` (parametrized, smaller scripts).
- [x] Failure-path e2e: uncalibrated judge blocks optimize+gate; budget exhaustion mid-GEPA still yields best-so-far + resumable checkpoint; gate rejects null candidates (no promotion).
- [x] CLI e2e: `init → calibrate → optimize → gate → registry activate → serve`(TestClient) all via CliRunner with fake scripts.
- [x] Commit `test: offline e2e suite`.

### Task 33: Dashboard e2e (Playwright)

- [x] `web/e2e/dashboard.spec.ts`: start FastAPI with seeded fixture registry+run (a committed `runs/` fixture with events.jsonl); assert Runs page renders curve, Lineage node click shows diff, Gate page promote button calls API (route-intercepted), Registry shows active badge. `npm run e2e` in CI-able form (chromium only).
- [x] Commit `test: dashboard playwright e2e`.

### Task 34: Live integration smoke (opt-in)

- [x] `tests/integration/test_live_smoke.py` marked `@pytest.mark.live` (skipped without `OPENROUTER_API_KEY`): 5 Bitext examples, judge on haiku, GEPA budget 20 rollouts, assert run completes, cost < $0.50 recorded, gate produces a report (either verdict).
- [x] `make live-smoke` target; document in CONTRIBUTING.
- [x] Commit `test: live integration smoke`.

### Task 35: Final verification sweep

- [x] `uv run pytest` full suite green; `ruff check`, `pyright` clean; `npm run build && npm run test` green; run offline demo end-to-end once via CLI; update plan checkboxes; final commit.

---

## Self-review notes
- Spec coverage: all spec sections map to tasks (core→2-8, judge→15-17, optimizers→10,11,19-22,27, gate/registry→23-24, serving→25, interfaces→12,26,28,29, demo→30, docs→31, testing→32-35). SPO explicitly deferred (spec lists it as later addition).
- Type consistency anchors: `Candidate`, `MetricResult`, `EvalReport`, `RunEvent`, `GateReport`, `CalibrationCertificate` defined once (Tasks 2,7,9,24,17) and referenced by name elsewhere.
- No placeholder steps: each task states files, behavior, test cases, and exact algorithms (GEPA Alg 1/2, merge triplet, Holm, bootstrap) inline or by precise reference to the spec's research citations.
