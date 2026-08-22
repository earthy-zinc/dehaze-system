"""add agent_version column to sys_ai_conversation

Revision ID: add_conv_agent_version
Revises: add_pred_eval_log_status
Create Date: 2026-08-18

会话版本锚定：为 sys_ai_conversation 新增 agent_version 字段，记录创建/切换
会话时锚定的 Agent 已发布版本号，运行面据此读取不可变快照组装推理图，
保证 Agent 发布/回滚不影响进行中会话（行为可复现）。
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_conv_agent_version'
down_revision = 'add_pred_eval_log_status'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 为 sys_ai_conversation 添加 agent_version 字段"""
    op.add_column(
        'sys_ai_conversation',
        sa.Column(
            'agent_version',
            sa.Integer,
            nullable=True,
            comment='会话锚定的Agent已发布版本号(创建/切换会话时写入,发布/回滚不影响进行中会话)',
        ),
    )


def downgrade():
    """Downgrade: 回滚 sys_ai_conversation.agent_version 字段"""
    op.drop_column('sys_ai_conversation', 'agent_version')
