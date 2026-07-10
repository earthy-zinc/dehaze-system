"""add size_bytes column to sys_file

Revision ID: add_file_size_bytes
Revises: add_operation_log
Create Date: 2026-02-20

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_file_size_bytes'
down_revision = 'add_operation_log'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 为 sys_file 表添加 size_bytes 列"""
    op.add_column(
        'sys_file',
        sa.Column(
            'size_bytes',
            sa.BigInteger(),
            nullable=False,
            server_default='0',
            comment='文件大小(字节)',
        ),
    )


def downgrade():
    """Downgrade: 移除 sys_file 表的 size_bytes 列"""
    op.drop_column('sys_file', 'size_bytes')
