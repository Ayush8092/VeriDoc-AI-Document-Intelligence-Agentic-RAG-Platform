# syntax=docker/dockerfile:1
#
# Phase 5 completion pass, round 2, item 15 ("Docker + Docker Compose").
#
# Three stages: install deps -> build -> minimal runtime, using
# Next.js's `output: "standalone"` (see next.config.ts) so the final
# image ships only the server + traced dependencies, not the full
# node_modules tree.

FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# --- build stage ---------------------------------------------------------

FROM node:22-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* variables are baked into the client bundle at BUILD time
# (Next.js's own convention, not specific to this project) — unlike the
# backend's runtime env vars, changing NEXT_PUBLIC_API_BASE_URL after
# this image is built requires rebuilding it, not just restarting the
# container. Accepted as a build ARG so docker-compose.yml /
# `docker build --build-arg` can set it per-environment without editing
# this file.
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

RUN npm run build

# --- runtime stage ---------------------------------------------------------

FROM node:22-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production

RUN useradd --create-home --uid 1001 veridoc

# Next.js standalone output: a self-contained server.js plus only the
# node_modules it actually traced as used at runtime — see
# https://nextjs.org/docs/app/api-reference/config/next-config-js/output
COPY --from=builder --chown=veridoc:veridoc /app/.next/standalone ./
COPY --from=builder --chown=veridoc:veridoc /app/.next/static ./.next/static
COPY --from=builder --chown=veridoc:veridoc /app/public ./public

USER veridoc

EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD node -e "fetch('http://localhost:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
