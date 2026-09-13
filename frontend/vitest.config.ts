import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

/**
 * Real automated frontend tests (previously: zero — see
 * tests/*.test.tsx for what's covered and why these specific surfaces
 * were picked first). Vitest + React Testing Library: the lowest-
 * friction choice for a Next.js App Router project — no separate
 * Babel/webpack config to maintain, and `vite-tsconfig-paths`-free `@/*`
 * resolution via the `resolve.alias` below (kept in sync with
 * tsconfig.json's own `paths` — see that file).
 */
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    css: false,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
});