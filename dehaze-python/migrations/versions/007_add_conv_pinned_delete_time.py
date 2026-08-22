"""add pinned_at and delete_time columns to sys_ai_conversation

Revision ID: 007_add_conv_pinned_delete_time
Revises: 006_add_billing_attribution
Create Date: 2026-08-18

会话生命周期补列：为 sys_ai_conversation 新增
- pinned_at：置顶时间（置顶会话按此倒序）
- delete_time：软删时间（30 天恢复窗口判定，超期由定时任务物理清理）
支撑批量操作/回收站/置顶上限/物理清理等生命周期能力。
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '007_add_conv_pinned_delete_time'
down_revision = '006_add_billing_attribution'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 为 sys_ai_conversation 添加置顶时间与软删时间字段"""
    op.add_column(
        'sys_ai_conversation',
        sa.Column(
            'pinned_at',
            sa.DateTime,
            nullable=True,
            comment='置顶时间(置顶会话按此倒序)',
        ),
    )
    op.add_column(
        'sys_ai_conversation',
        sa.Column(
            'delete_time',
            sa.DateTime,
            nullable=True,
            comment='软删时间(30天恢复窗口判定，超期由定时任务物理清理)',
        ),
    )
    # 置顶索引补充置顶时间列以支持按置顶时间倒序（原 idx_user_pinned(user_id,pinned) 重建）；
    # 新增软删时间索引供回收站查询与物理清理扫描
    op.drop_index('idx_user_pinned', table_name='sys_ai_conversation')
    op.create_index(
        'idx_user_pinned',
        'sys_ai_conversation',
        ['user_id', 'pinned', 'pinned_at'],
        unique=False,
    )
    op.create_index(
        'idx_deleted_time',
        'sys_ai_conversation',
        ['deleted', 'delete_time'],
        unique=False,
    )


def downgrade():
    """Downgrade: 回滚 sys_ai_conversation 置顶时间与软删时间字段"""
    op.drop_index('idx_deleted_time', table_name='sys_ai_conversation')
    op.drop_index('idx_user_pinned', table_name='sys_ai_conversation')
    op.drop_column('sys_ai_conversation', 'delete_time')
    op.drop_column('sys_ai_conversation', 'pinned_at')
