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
