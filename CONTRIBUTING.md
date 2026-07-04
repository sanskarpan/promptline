# Contributing to Promptline

## Setup

```sh
uv sync                 # Python 3.11+, installs promptline + dev deps
cd web && npm install   # only needed for dashboard work
```

## Test tiers

| Tier | Command | Needs | When |
|---|---|---|---|
| Offline suite (unit + e2e) | `make test` (`uv run pytest -q`) | nothing | every change |
| Dashboard e2e | `make e2e` | `npm run build` in `web/` + `npx playwright install chromium` | dashboard changes |
| Live smoke | `make live-smoke` | `OPENROUTER_API_KEY` (< $0.50) | before releases / adapter changes |

The offline suite is the gate for merging; the live tier is opt-in and marked
`@pytest.mark.live` (auto-skipped without a key). Lint with `make lint`.

## The fake-script pattern

Tests never hit a real LLM. Two mechanisms:

1. **In Python** — `FakeLLMClient(script=...)` accepts a list of canned
   responses or a callable `(LLMCall) -> str`. The canonical e2e trick
   (see `tests/e2e/conftest.py`): the client answers with a marker string
   iff the system prompt already contains it, while scripted
   reflection/proposal calls return an instruction that *adds* the marker —
   so a metric that rewards the marker makes optimizers genuinely "improve".
2. **Through the CLI** — set `PROMPTLINE_FAKE_SCRIPT=/path/to/script.json`.
   The JSON holds a cycling `"responses": [...]` list and/or first-match
   `"keyed": [{"contains": ..., "response": ...}]` rules matched against the
   full prompt text. See `tests/e2e/test_cli_e2e.py` for a complete
   init → calibrate → optimize → gate → serve chain driven this way.

## Conventions

- Conventional commits (`feat:`, `fix:`, `test:`, `docs:` …).
- `uv run ruff check .` must pass; keep new tests deterministic (seeded RNG,
  scripted clients, tmp_path workspaces).
- Prefer composing real components in tests and faking only the LLM.
