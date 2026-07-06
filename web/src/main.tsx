import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";
import "./theme.css";
import { Layout } from "./Layout";
import { RunsPage } from "./pages/RunsPage";
import { RunDetail } from "./pages/RunDetail";
import { LineagePage } from "./pages/LineagePage";
import { JudgePage } from "./pages/JudgePage";
import { GatePage } from "./pages/GatePage";
import { RegistryPage } from "./pages/RegistryPage";
import { NotFound } from "./pages/NotFound";

// All SPA pages live under /ui/... so hard refreshes on those paths hit the
// static fallback (index.html) rather than the API's GET /runs endpoints.
const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Navigate to="/ui/runs" replace /> },
      { path: "ui/runs", element: <RunsPage /> },
      { path: "ui/runs/:id", element: <RunDetail /> },
      { path: "ui/lineage/:runId", element: <LineagePage /> },
      { path: "ui/judge", element: <JudgePage /> },
      { path: "ui/gate", element: <GatePage /> },
      { path: "ui/registry", element: <RegistryPage /> },
      // Catch-all: keep the sidebar/nav and offer a way back instead of the
      // bare default error boundary.
      { path: "*", element: <NotFound /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
