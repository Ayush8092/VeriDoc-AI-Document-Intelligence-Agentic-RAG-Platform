#!/bin/sh
set -e

# Phase 5 completion pass, round 2, item 15. Runs once per container
# start, before exec'ing the real CMD (uvicorn, or a one-shot command
# like `alembic upgrade head` itself for the dedicated `migrate` service
# in docker-compose.yml — see that file).
#
# RUN_MIGRATIONS_ON_START=1 is OFF by default deliberately: in a
# multi-replica deployment, every replica running `alembic upgrade head`
# concurrently on container start is a real race (two replicas racing
# to apply the same migration). docker-compose.yml's `migrate` service
# runs migrations exactly once, as a dependency the `backend`/`worker`
# services wait on (`depends_on: migrate: condition: service_completed_successfully`),
# which is the correct pattern for more than one replica. This flag
# exists for a genuinely single-replica/local-dev setup where that race
# can't happen, as a convenience.
if [ "${RUN_MIGRATIONS_ON_START:-0}" = "1" ]; then
    echo "[entrypoint] RUN_MIGRATIONS_ON_START=1 — running 'alembic upgrade head'..."
    alembic upgrade head
fi

exec "$@"
