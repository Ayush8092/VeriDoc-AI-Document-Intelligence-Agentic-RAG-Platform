"""Database engine/session setup.

Uses SQLAlchemy so the same models work against SQLite (zero-setup local
dev and tests) and PostgreSQL (production — set `DATABASE_URL`, see
docs/architecture.md "PostgreSQL"). This module owns *connecting*; the
`documents`/`services` layers own what's done with the session.

Known limitation (tracked in MIGRATION_PLAN.md): schema migrations use
`Base.metadata.create_all` rather than Alembic. That's adequate for this
phase (no destructive schema changes shipped yet) but a real deployment
needs versioned migrations before its second schema change — see
docs/architecture.md, "Limitations".
"""

from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal: sessionmaker | None = None


def _build_engine(settings: Settings):
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(settings.database_url, connect_args=connect_args, future=True)


def get_engine(settings: Settings | None = None):
    global _engine
    if _engine is None:
        _engine = _build_engine(settings or get_settings())
    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(settings), autoflush=False, autocommit=False, future=True)
    return _SessionLocal


def init_db(settings: Settings | None = None) -> None:
    """Create all tables that don't exist yet. Safe to call repeatedly."""
    from app.db import models  # noqa: F401  (register models on Base.metadata)

    Base.metadata.create_all(bind=get_engine(settings))


@contextmanager
def session_scope(settings: Settings | None = None):
    """Provide a transactional scope around a series of operations."""
    factory = get_session_factory(settings)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_for_tests() -> None:
    """Drop cached engine/session-factory so tests can point at a fresh DB."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
