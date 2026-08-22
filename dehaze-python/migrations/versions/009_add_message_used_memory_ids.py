"""add used_memory_ids column to sys_ai_message

Revision ID: 009_add_message_used_memory_ids
Revises: 008_add_conv_summary_watermark
Create Date: 2026-08-19

记忆注入可见性：为 sys_ai_message 新增
- used_memory_ids：本条助手消息本次注入引用的记忆 ID 列表（JSON 数组），
  支撑"注入可见性"——用户可展开查看每条回复引用了哪些长期记忆。

由 memory 域在推理层写入（inject_memories 返回的 injected_list），
供前端按记忆 ID 展示本次引用的记忆清单。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import JSON


# revision identifiers, used by Alembic.
revision = '009_add_message_used_memory_ids'
down_revision = '008_add_conv_summary_watermark'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 为 sys_ai_message 添加注入记忆引用字段"""
    op.add_column(
        'sys_ai_message',
        sa.Column(
            'used_memory_ids',
            JSON,
            nullable=True,
            comment='本次注入引用的记忆ID列表(JSON数组,注入可见性)',
        ),
    )


def downgrade():
    """Downgrade: 回滚 sys_ai_message 注入记忆引用字段"""
    op.drop_column('sys_ai_message', 'used_memory_ids')
