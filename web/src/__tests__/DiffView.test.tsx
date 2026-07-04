import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DiffView } from "../components/DiffView";

describe("DiffView", () => {
  it("renders insert and delete spans", () => {
    render(<DiffView before="answer the question briefly" after="answer the question precisely" />);
    const view = screen.getByTestId("diff-view");
    const del = view.querySelector(".diff-del");
    const ins = view.querySelector(".diff-ins");
    expect(del).not.toBeNull();
    expect(ins).not.toBeNull();
    expect(del!.textContent).toContain("briefly");
    expect(ins!.textContent).toContain("precisely");
  });

  it("renders unchanged text without diff classes", () => {
    render(<DiffView before="same text" after="same text" />);
    const view = screen.getByTestId("diff-view");
    expect(view.querySelector(".diff-del")).toBeNull();
    expect(view.querySelector(".diff-ins")).toBeNull();
    expect(view.textContent).toBe("same text");
  });
});
