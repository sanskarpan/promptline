# Promptline Web Dashboard

Terminal-styled React dashboard for the Promptline control plane: live run
monitoring (SSE), candidate lineage with diffs, judge calibration
certificates, statistical gating and prompt registry management.

## Development

Run the API and the dashboard side by side:

```sh
# terminal 1 — API on :8000
uv run promptline serve

# terminal 2 — dashboard on :5173 (proxies /runs, /gate, ... to :8000)
cd web
npm install
npm run dev
```

## Production

Build once, then FastAPI serves the bundle itself:

```sh
cd web
npm run build          # emits web/dist/

uv run promptline serve   # dashboard now available at http://127.0.0.1:8000/
```

`promptline serve` mounts `web/dist/` at `/` when it exists; API routes take
precedence and unknown non-API paths fall back to `index.html` (SPA routing).

Set `VITE_API_BASE` at build time to point the client at a non-origin API.

## Tests

```sh
npm test   # vitest run — reducer, lineage layout, diff view
```

## End-to-end (Playwright)

The e2e suite drives the built dashboard against a seeded fixture API
(`tests/e2e/serve_fixture.py`: a finished GEPA run, an ACTIVE gated prompt and
a passing judge certificate). Playwright starts and stops the server itself.

```sh
cd web
npm run build                                # e2e serves web/dist
npx playwright install chromium --with-deps  # or: npx playwright install chromium
npm run e2e
```

Notes:

- Chromium-only project, `retries: 0` — failures are real, never flaky-retried.
- Requires a browser download; in environments where Chromium cannot be
  installed (no network / unsupported OS), skip this suite — it is not part of
  `uv run pytest` and CI should treat it as a separate opt-in job.
- The fixture server binds `127.0.0.1:8788` (override with
  `PROMPTLINE_FIXTURE_PORT`, mirrored in `playwright.config.ts`).
