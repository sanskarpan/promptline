import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { CalibrationCertificate } from "../api";

const certificates = vi.fn();
vi.mock("../api", () => ({
  api: { certificates: () => certificates() },
}));

import { JudgePage } from "../pages/JudgePage";

function cert(overrides: Partial<CalibrationCertificate>): CalibrationCertificate {
  return {
    judge_name: "helpfulness",
    criterion: "overall",
    kappa: 1,
    spearman: 1,
    n_holdout: 10,
    threshold: 0.6,
    passed: true,
    degenerate: false,
    confusion: [[1]],
    label_min: 0,
    label_max: 1,
    created_at: "2026-01-01",
    ...overrides,
  };
}

describe("JudgePage", () => {
  beforeEach(() => certificates.mockReset());

  it("renders '—' instead of 'NaN' for a degenerate spearman", async () => {
    certificates.mockResolvedValue([
      cert({ degenerate: true, spearman: NaN, kappa: NaN }),
    ]);
    render(<JudgePage />);

    // Find the Spearman stat card and assert it does not read "NaN".
    const spearmanCard = await waitFor(() => {
      const label = screen
        .getAllByText("Spearman")
        .find((el) => el.classList.contains("k"));
      expect(label).toBeTruthy();
      return label!.parentElement!;
    });
    expect(spearmanCard.textContent).not.toContain("NaN");
    expect(spearmanCard.querySelector(".v")!.textContent).toBe("—");
  });

  it("still formats a finite spearman numerically", async () => {
    certificates.mockResolvedValue([cert({ spearman: 0.5 })]);
    render(<JudgePage />);
    await waitFor(() => {
      const label = screen
        .getAllByText("Spearman")
        .find((el) => el.classList.contains("k"));
      expect(label!.parentElement!.querySelector(".v")!.textContent).toBe(
        "0.500",
      );
    });
  });
});
