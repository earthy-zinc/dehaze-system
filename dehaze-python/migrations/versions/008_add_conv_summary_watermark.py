"""add summary_upto_message_id column to sys_ai_conversation

Revision ID: 008_add_conv_summary_watermark
Revises: 007_add_conv_pinned_delete_time
Create Date: 2026-08-19

摘要增量治理：为 sys_ai_conversation 新增摘要水位字段
- summary_upto_message_id：摘要覆盖到的最后一条消息 ID，用于增量摘要
  （仅摘要"上次摘要覆盖位置之后、最近 N 轮之前"的消息，避免全量重摘导致摘要无限膨胀）。
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '008_add_conv_summary_watermark'
down_revision = '007_add_conv_pinned_delete_time'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 为 sys_ai_conversation 添加摘要水位字段"""
    op.add_column(
        'sys_ai_conversation',
        sa.Column(
            'summary_upto_message_id',
            sa.BigInteger,
            nullable=True,
            comment='摘要水位：已纳入摘要覆盖范围的最后一条消息ID(增量摘要推进依据)',
        ),
    )


def downgrade():
    """Downgrade: 回滚 sys_ai_conversation 摘要水位字段"""
    op.drop_column('sys_ai_conversation', 'summary_upto_message_id')
