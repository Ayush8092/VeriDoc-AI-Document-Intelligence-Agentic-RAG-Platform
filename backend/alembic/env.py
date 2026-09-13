"""Alembic migration environment for Veridoc (Phase 5).

Reads `DATABASE_URL` the SAME WAY the running application does
(`app.core.config.get_settings`) rather than a separate, easy-to-drift
value baked into `alembic.ini` — the one thing worse than no migrations
is migrations that silently target a different database than the app
actually runs against. `alembic.ini`'s own `sqlalchemy.url` is left as a
placeholder (see that file's comment) precisely so this is the only
place the real URL is read from.

`init_db()` (`app/db/session.py`, `Base.metadata.create_all`) remains
the zero-setup path for local dev and the test suite — every existing
test fixture already relies on it, and changing that now would be a
large, risky, untested change for a project this size. Alembic is the
path for a real deployment: `alembic upgrade head` against a fresh
PostgreSQL database, and every future schema change from here on gets
its own migration instead of a silent `create_all` change (see
docs/architecture.md, "Database migrations").
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.db.session import Base  # noqa: E402
from app.db import models  # noqa: E402,F401  (registers every table on Base.metadata)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Override alembic.ini's placeholder URL with the app's real, live
# configuration (env vars / .env — see app.core.config.Settings).
config.set_main_option("sqlalchemy.url", get_settings().database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()