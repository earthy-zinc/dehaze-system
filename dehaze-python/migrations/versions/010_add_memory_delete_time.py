"""add delete_time column to sys_ai_memory

Revision ID: 010_add_memory_delete_time
Revises: 009_add_message_used_memory_ids
Create Date: 2026-08-19

记忆生命周期补列：为 sys_ai_memory 新增
- delete_time：软删时间（批量清空/单条删除时记录，30 天恢复窗口判定，
  超期由定时任务 purgeDeletedMemories 物理清理）

与会话域 delete_time 模式一致（见 007_add_conv_pinned_delete_time）。
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '010_add_memory_delete_time'
down_revision = '009_add_message_used_memory_ids'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 为 sys_ai_memory 添加软删时间字段"""
    op.add_column(
        'sys_ai_memory',
        sa.Column(
            'delete_time',
            sa.DateTime,
            nullable=True,
            comment='软删时间(30天恢复窗口判定，超期由定时任务物理清理)',
        ),
    )
    op.create_index(
        'idx_deleted_time',
        'sys_ai_memory',
        ['deleted', 'delete_time'],
        unique=False,
    )


def downgrade():
    """Downgrade: 回滚 sys_ai_memory 软删时间字段"""
    op.drop_index('idx_deleted_time', table_name='sys_ai_memory')
    op.drop_column('sys_ai_memory', 'delete_time')
