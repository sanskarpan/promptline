import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PORT = 8788;
const BASE_URL = `http://127.0.0.1:${PORT}`;

/**
 * Dashboard e2e against a seeded fixture server (tests/e2e/serve_fixture.py):
 * a finished GEPA run, a registry with an active gated prompt, and a passing
 * judge certificate. Requires `npm run build` (web/dist) and a chromium
 * install (`npx playwright install chromium`) — see web/README.md.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "uv run python -m tests.e2e.serve_fixture",
    cwd: path.resolve(__dirname, ".."),
    url: `${BASE_URL}/runs`,
    env: { PROMPTLINE_FIXTURE_PORT: String(PORT) },
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
