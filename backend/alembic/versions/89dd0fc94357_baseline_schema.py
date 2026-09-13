"""baseline schema

Revision ID: 89dd0fc94357
Revises: 
Create Date: 2026-08-18 01:50:28.442349

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '89dd0fc94357'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Hand-adjusted from the raw autogenerate output (see this migration's
    generation log): `documents` and `document_versions` reference each
    other (`documents.current_version_id -> document_versions.id`,
    `document_versions.document_id -> documents.id`), which alembic's
    autogenerate can't topologically order (it warned "unresolvable
    cycles" and silently dropped ONE of the two FK constraints from
    table-creation order, creating `document_versions` — with a live FK
    to a `documents` table that didn't exist yet — before `documents`
    itself). That ordering happens to work on SQLite (foreign keys are
    unenforced by default) but would fail outright on PostgreSQL with
    "relation \"documents\" does not exist". Fixed the standard way for a
    genuine FK cycle: create both tables with the forward-reference
    column present but NOT constrained, then add both foreign key
    constraints afterward once both tables exist.
    """
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('hashed_password', sa.String(length=128), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    op.create_table('documents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('source_root', sa.String(length=32), nullable=False),
    sa.Column('filename', sa.String(length=512), nullable=False),
    sa.Column('file_type', sa.String(length=16), nullable=False),
    sa.Column('title', sa.String(length=512), nullable=False),
    sa.Column('owner_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('current_version_id', sa.Integer(), nullable=True),  # FK added below, once document_versions exists
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_documents_owner_id'), 'documents', ['owner_id'], unique=False)

    op.create_table('document_versions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('version_number', sa.Integer(), nullable=False),
    sa.Column('file_hash', sa.String(length=64), nullable=False),
    sa.Column('page_count', sa.Integer(), nullable=False),
    sa.Column('has_scanned_pages', sa.Boolean(), nullable=False),
    sa.Column('table_count', sa.Integer(), nullable=False),
    sa.Column('chunk_count', sa.Integer(), nullable=False),
    sa.Column('mean_ocr_confidence', sa.Float(), nullable=True),
    sa.Column('warnings', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_document_versions_file_hash'), 'document_versions', ['file_hash'], unique=False)

    # Now that both sides of the cycle exist, add the second FK.
    with op.batch_alter_table('documents') as batch_op:
        batch_op.create_foreign_key(
            'fk_documents_current_version_id', 'document_versions', ['current_version_id'], ['id']
        )

    op.create_table('ingestion_runs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('reset', sa.Boolean(), nullable=False),
    sa.Column('chunks_total', sa.Integer(), nullable=False),
    sa.Column('chunks_changed', sa.Integer(), nullable=False),
    sa.Column('vectors_upserted', sa.Integer(), nullable=False),
    sa.Column('stale_vectors_deleted', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('detail', sa.Text(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('ingestion_runs')
    with op.batch_alter_table('documents') as batch_op:
        batch_op.drop_constraint('fk_documents_current_version_id', type_='foreignkey')
    op.drop_index(op.f('ix_document_versions_file_hash'), table_name='document_versions')
    op.drop_table('document_versions')
    op.drop_index(op.f('ix_documents_owner_id'), table_name='documents')
    op.drop_table('documents')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')