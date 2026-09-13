import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Phase 5 completion pass, round 2, item 15 (Docker): "standalone"
  // produces a minimal, self-contained `.next/standalone` output
  // (server + only the node_modules actually used at runtime traced in)
  // instead of requiring the full `node_modules` tree in the final
  // image — see frontend/Dockerfile's multi-stage build, which copies
  // exactly that output into the runtime stage.
  output: "standalone",
};

export default nextConfig;
