import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// @testing-library/react only auto-cleans up between tests under Jest's
// global afterEach hook; Vitest needs this wired explicitly, or a
// previous test's rendered tree (and its running effects/timers) leaks
// into the next test's `document.body` — exactly what caused every
// AuthProvider test after the first to see multiple stacked <div> trees
// and match the wrong one via getByTestId.
afterEach(() => {
  cleanup();
});