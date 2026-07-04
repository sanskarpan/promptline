.PHONY: test e2e live-smoke lint

## Offline test suite (fast, no network, no API key).
test:
	uv run pytest -q

## Dashboard Playwright e2e (needs `npm run build` + chromium; see web/README.md).
e2e:
	cd web && npm run e2e

## Opt-in live smoke against OpenRouter (needs OPENROUTER_API_KEY; ~<$0.50).
live-smoke:
	uv run pytest -m live tests/integration -q

lint:
	uv run ruff check .
