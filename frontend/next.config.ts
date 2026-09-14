import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Phase 5 completion pass, round 2, item 15 (Docker): "standalone"
  // produces a minimal, self-contained `.next/standalone` output
  // (server + only the node_modules actually used at runtime traced in)
  // instead of requiring the full `node_modules` tree in the final
  // image — see frontend/Dockerfile's multi-stage build, which copies
  // exactly that output into the runtime stage.
  //
  // Vercel deploys must NOT use "standalone" — Vercel's own build
  // pipeline generates its own serverless function file-trace
  // manifests (e.g. next-server.js.nft.json) and expects the default
  // output format; forcing "standalone" breaks that step with an
  // ENOENT on the .nft.json file. Vercel sets the VERCEL env var
  // during its builds, so use that to apply "standalone" only for the
  // Docker build (where VERCEL is unset) and leave Vercel's own builds
  // on the default output.
  output: process.env.VERCEL ? undefined : "standalone",
};

export default nextConfig;