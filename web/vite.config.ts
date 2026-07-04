import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    proxy: {
      "/prompts": backend,
      "/runs": backend,
      "/gate": backend,
      "/registry": backend,
      "/judges": backend,
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
    globals: true,
    // Playwright specs live in e2e/ and must not run under vitest.
    exclude: ["e2e/**", "node_modules/**"],
  },
});
