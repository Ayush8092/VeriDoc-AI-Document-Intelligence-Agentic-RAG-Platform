"""add ingestion_jobs table

Revision ID: f3a1c9d2e7b4
Revises: 89dd0fc94357
Create Date: 2026-08-18 00:00:00.000000

Phase 5 completion pass, item 3.2: `app/db/models.py::IngestionJob` has
existed since the background-ingestion job engine
(`app/services/ingestion_jobs.py`) was written, but no migration ever
created its table — a real, confirmed gap (see
`veridoc_phase5_status_and_continuation_prompt.docx`, section 1). This
migration creates exactly the columns/indexes currently on that model,
nothing more/less; if the model changes again, regenerate rather than
hand-editing both out of sync.

Straightforward addition, no FK-cycle complication like the baseline
migration had: `ingestion_jobs` only references EXISTING tables
(`documents`, `document_versions`, `users`), all created by
`89dd0fc94357` before this revision runs.

Verified (not merely written) in a later pass, once alembic/sqlalchemy
were actually installable: `alembic upgrade head`, `alembic downgrade
base`, and `alembic check` (which diffs live SQLAlchemy model metadata
against the migration chain and reports zero drift) all pass clean
against a fresh SQLite database.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3a1c9d2e7b4'
down_revision: Union[str, Sequence[str], None] = '89dd0fc94357'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'ingestion_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=True),
        sa.Column('version_id', sa.Integer(), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=True),
        sa.Column('filename', sa.String(length=512), nullable=False),
        sa.Column('source_root', sa.String(length=32), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('attempt_count', sa.Integer(), nullable=False),
        sa.Column('max_attempts', sa.Integer(), nullable=False),
        sa.Column('error', sa.Text(), nullable=False),
        sa.Column('chunks_total', sa.Integer(), nullable=False),
        sa.Column('chunks_changed', sa.Integer(), nullable=False),
        sa.Column('vectors_upserted', sa.Integer(), nullable=False),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id']),
        sa.ForeignKeyConstraint(['version_id'], ['document_versions.id']),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_ingestion_jobs_document_id'), 'ingestion_jobs', ['document_id'], unique=False)
    op.create_index(op.f('ix_ingestion_jobs_owner_id'), 'ingestion_jobs', ['owner_id'], unique=False)
    op.create_index(op.f('ix_ingestion_jobs_status'), 'ingestion_jobs', ['status'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_ingestion_jobs_status'), table_name='ingestion_jobs')
    op.drop_index(op.f('ix_ingestion_jobs_owner_id'), table_name='ingestion_jobs')
    op.drop_index(op.f('ix_ingestion_jobs_document_id'), table_name='ingestion_jobs')
    op.drop_table('ingestion_jobs')
