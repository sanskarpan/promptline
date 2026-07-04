# Registry and serving

## Registry

`promptline.registry.registry.PromptRegistry` is a thread-safe SQLite store at `<registry.path>/registry.db` with four tables:

- **prompts** — every registered `Candidate` (full JSON), its program, run id, and `parent_ids` (lineage);
- **evals** — append-only eval history (dataset hash, mean score, n);
- **active** — one row per program: the currently served prompt;
- **activation_history** — every pointer move, as `activate` / `rollback` actions.

Key invariants:

- `register()` is idempotent on the candidate id; `optimize` auto-registers its best candidate.
- `activate()` is the **only** method that moves the pointer forward, and it stores the gate report JSON alongside. The CLI's `registry activate` exists for bootstrapping a baseline and prints a warning; gated promotions come from `promptline gate`.
- `rollback()` replays `activation_history` as an undo stack (consecutive duplicate activations compressed, each rollback pops) and reverts to the previous *distinct* prompt — `RuntimeError` when there is none.
- `lineage(prompt_id)` walks `parent_ids` breadth-first — GEPA's evolution tree falls out of this naturally.

CLI: `promptline registry list | show <id> | activate <id> | rollback`.

## Server

`promptline serve` builds one FastAPI app (`promptline.server.app.create_app`) with two planes plus the static dashboard.

### Serving plane

`GET /prompts/{program}/active` is the production deploy target apps poll:

- 404 when the program has no active prompt;
- otherwise the module instructions/demos, prompt id, and activation timestamp;
- **ETag** = the prompt id (quoted per RFC 7232). Poll with `If-None-Match: "<id>"` and get an empty `304` until the gate promotes something new — a deploy is just the ETag changing.

```bash
curl -i http://127.0.0.1:8000/prompts/support/active
curl -i -H 'If-None-Match: "<prompt_id>"' http://127.0.0.1:8000/prompts/support/active   # 304
```

### Control plane

| Method & path | Purpose |
|---|---|
| `POST /runs` | Start an optimizer run (`{optimizer, data_path?, budget?}`); returns `run_id`; synchronous setup errors → 400 with the failed run id |
| `GET /runs` / `GET /runs/{id}` | List runs / one run's status |
| `GET /runs/{id}/events` | SSE stream tailing the run's `events.jsonl` (live) or replaying it (finished) |
| `POST /gate` | Run the deploy gate (`{program, candidate_ids, dev_path, val_path, incumbent_id?}`); returns the `GateReport`; refusals → 400 |
| `GET /registry/{program}` | Registered prompts + latest mean scores |
| `POST /registry/{program}/activate` | Move the active pointer (`{prompt_id, gate_report?}`) |
| `POST /registry/{program}/rollback` | Undo the last distinct activation (409 when impossible) |
| `GET /judges/certificates` | All saved calibration certificates |

The app is dependency-injected: the CLI supplies `run_starter` and `gate_runner` closures built from `promptline.yaml`; tests inject fakes via FastAPI's TestClient.

### Dashboard

When `web/dist/index.html` exists, it is mounted at `/` *after* all API routes (API paths always win) with SPA fallback routing. The TUI (`promptline tui --run <id>` or `--attach <sse-url>`) consumes the same event stream.
