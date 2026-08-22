"""add summary column to sys_ai_agent_thought

Revision ID: 011_add_agent_thought_summary
Revises: 010_add_memory_delete_time
Create Date: 2026-08-19

多步推理两级展示（§5.4）：为 sys_ai_agent_thought 新增 summary 列，
存 LLM 对每条推理步骤的一句话概括（一级步骤摘要，synthesize_response
阶段异步生成，不阻塞主回复）。
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '011_add_agent_thought_summary'
down_revision = '010_add_memory_delete_time'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 为 sys_ai_agent_thought 添加步骤摘要字段"""
    op.add_column(
        'sys_ai_agent_thought',
        sa.Column(
            'summary',
            sa.Text,
            nullable=True,
            comment='步骤一句话摘要(LLM生成,两级展示一级:步骤摘要)',
        ),
    )


def downgrade():
    """Downgrade: 回滚 sys_ai_agent_thought 步骤摘要字段"""
    op.drop_column('sys_ai_agent_thought', 'summary')
