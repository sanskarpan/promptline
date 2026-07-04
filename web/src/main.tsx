import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./theme.css";
import { Layout } from "./Layout";
import { RunsPage } from "./pages/RunsPage";
import { RunDetail } from "./pages/RunDetail";
import { LineagePage } from "./pages/LineagePage";
import { JudgePage } from "./pages/JudgePage";
import { GatePage } from "./pages/GatePage";
import { RegistryPage } from "./pages/RegistryPage";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <RunsPage /> },
      { path: "runs", element: <RunsPage /> },
      { path: "runs/:id", element: <RunDetail /> },
      { path: "lineage/:runId", element: <LineagePage /> },
      { path: "judge", element: <JudgePage /> },
      { path: "gate", element: <GatePage /> },
      { path: "registry", element: <RegistryPage /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
