import { expect, test } from "@playwright/test";

/**
 * The fixture server (tests/e2e/serve_fixture.py) seeds:
 * - run "fixture-run": finished GEPA run, seed (score 0) -> winner (score 1),
 *   winner instruction contains "ALWAYS CITE SOURCES";
 * - registry program "support": 2 prompts, winner ACTIVE with a gate report;
 * - one passing "helpfulness" calibration certificate.
 */

const RUN_ID = "fixture-run";
const MARKER = "ALWAYS CITE SOURCES";

test("runs page lists the finished fixture run", async ({ page }) => {
  await page.goto("/ui/runs");
  const row = page.locator("tr", { hasText: RUN_ID });
  await expect(row).toBeVisible();
  await expect(row).toContainText("FINISHED");
  await expect(row).toContainText("1.0000"); // best score
});

test("run detail renders score curve, budget meters and event feed", async ({
  page,
}) => {
  await page.goto(`/ui/runs/${RUN_ID}`);
  // Stat cards fed by the replayed SSE stream.
  await expect(page.locator(".stat-card", { hasText: "Status" })).toContainText(
    "FINISHED",
  );
  await expect(page.getByTestId("score-curve")).toBeVisible();
  // Budget meters (rollouts + cost) rendered from budget_tick events.
  await expect(page.locator(".meter-row", { hasText: "Rollouts" })).toBeVisible();
  await expect(page.locator(".meter-row", { hasText: "Cost USD" })).toBeVisible();
  // Event feed contains the bracketing lifecycle events.
  const feed = page.getByTestId("event-feed");
  await expect(feed).toContainText("run_started");
  await expect(feed).toContainText("full_eval");
  await expect(feed).toContainText("run_finished");
});

test("lineage node click opens instruction panel with diff", async ({ page }) => {
  await page.goto(`/ui/lineage/${RUN_ID}`);
  const svg = page.getByTestId("lineage-svg");
  await expect(svg).toBeVisible();

  const nodes = page.locator(".lineage-node");
  await expect(nodes.first()).toBeVisible();
  expect(await nodes.count()).toBeGreaterThanOrEqual(2);

  // Node 0 is the seed (full_eval only); node 1 is the proposed child whose
  // candidate_proposed payload carries the improved instruction.
  await nodes.nth(1).click();
  await expect(page.locator(".side")).toContainText("Instruction");
  await expect(page.locator(".side .mono-block").first()).toContainText(MARKER);
  await expect(page.locator(".side")).toContainText("Diff vs parent");
});

test("judge page shows PASS badge and confusion grid", async ({ page }) => {
  await page.goto("/ui/judge");
  await expect(page.locator(".badge", { hasText: "PASS" })).toBeVisible();
  await expect(page.getByTestId("confusion")).toBeVisible();
  await expect(page.locator(".stat-card", { hasText: "Kappa" })).toContainText(
    "1.000",
  );
});

test("registry page shows the ACTIVE badge for the gated winner", async ({
  page,
}) => {
  await page.goto("/ui/registry");
  // Labels are not htmlFor-wired; scope by the .field wrapper instead.
  await page
    .locator(".field", { hasText: "Program" })
    .locator("input")
    .fill("support");
  await page.getByRole("button", { name: "Load" }).click();

  await expect(page.locator("table tbody tr")).toHaveCount(2);
  const activeRow = page.locator("tr", { hasText: "ACTIVE" });
  await expect(activeRow).toBeVisible();
  await expect(activeRow).toContainText("1.0000"); // winner's mean score
});

test("gate page renders the gate form", async ({ page }) => {
  // Form render only: POST /gate needs dev/val dataset paths on the server
  // and the fixture server has no gate_runner wired, so no submission here.
  await page.goto("/ui/gate");
  for (const label of [
    "Program",
    "Incumbent ID (optional)",
    "Candidate IDs (comma/space sep)",
    "Dev path",
    "Val path",
  ]) {
    const field = page.locator(".field", { hasText: label });
    await expect(field).toBeVisible();
    await expect(field.locator("input")).toBeVisible();
  }
  await expect(page.getByRole("button", { name: /run gate/i })).toBeVisible();
});
