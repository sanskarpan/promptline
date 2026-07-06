.PHONY: help install test e2e live-smoke lint format format-check typecheck check web-test web-build build

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Sync all dependencies (runtime + data extra + dev)
	uv sync --all-extras --dev

test:  ## Offline test suite (fast, no network, no API key)
	uv run pytest -q

lint:  ## Ruff lint
	uv run ruff check .

format:  ## Apply Ruff formatting
	uv run ruff format .

format-check:  ## Check Ruff formatting without writing
	uv run ruff format --check .

typecheck:  ## Pyright type check
	uv run pyright promptline

check: lint format-check typecheck test  ## Run the full local gate (matches CI python job)

web-test:  ## Dashboard unit tests (vitest)
	cd web && npm test

web-build:  ## Build the dashboard (writes web/dist)
	cd web && npm run build

e2e:  ## Dashboard Playwright e2e (needs web-build + chromium; see web/README.md)
	cd web && npm run e2e

live-smoke:  ## Opt-in live smoke against OpenRouter (needs OPENROUTER_API_KEY; ~<$0.50)
	uv run pytest -m live tests/integration -q

build:  ## Build the sdist + wheel
	uv build
