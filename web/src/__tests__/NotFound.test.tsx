import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { Layout } from "../Layout";
import { NotFound } from "../pages/NotFound";

/** Mirror the catch-all wiring from main.tsx without the API-fetching pages. */
function renderAt(path: string) {
  const router = createMemoryRouter(
    [
      {
        path: "/",
        element: <Layout />,
        children: [
          { index: true, element: <div>home</div> },
          { path: "*", element: <NotFound /> },
        ],
      },
    ],
    { initialEntries: [path] },
  );
  render(<RouterProvider router={router} />);
}

describe("unknown /ui route", () => {
  it("keeps the sidebar/nav and offers a link back to Runs", () => {
    renderAt("/ui/bogus");
    // Sidebar brand + nav links stay visible (not the bare error boundary).
    expect(screen.getByText("PROMPTLINE")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Runs" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Judge" })).toBeInTheDocument();
    // Friendly not-found content with a way back.
    expect(screen.getByTestId("not-found")).toBeInTheDocument();
    const back = screen.getByRole("link", { name: /back to runs/i });
    expect(back).toHaveAttribute("href", "/ui/runs");
  });
});
