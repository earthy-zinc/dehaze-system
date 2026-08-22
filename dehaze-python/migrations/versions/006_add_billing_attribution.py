"""add cost attribution columns to sys_ai_billing

Revision ID: 006_add_billing_attribution
Revises: add_conv_agent_version
Create Date: 2026-08-18

成本归因：为 sys_ai_billing 新增 request_id / provider_id / error_code /
latency_ms 字段（对齐模型管理 §3.1 成本归因），支撑供应商健康聚合与成本对账。
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '006_add_billing_attribution'
down_revision = 'add_conv_agent_version'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 为 sys_ai_billing 添加成本归因字段"""
    op.add_column(
        'sys_ai_billing',
        sa.Column(
            'request_id',
            sa.String(64),
            nullable=True,
            comment='请求唯一ID(支撑对账与异常追溯)',
        ),
    )
    op.add_column(
        'sys_ai_billing',
        sa.Column(
            'provider_id',
            sa.BigInteger,
            nullable=True,
            comment='实际供应商ID(关联sys_ai_provider.id)',
        ),
    )
    op.add_column(
        'sys_ai_billing',
        sa.Column(
            'error_code',
            sa.String(16),
            nullable=True,
            comment='调用失败错误码(如429/5xx,成功为NULL)',
        ),
    )
    op.add_column(
        'sys_ai_billing',
        sa.Column(
            'latency_ms',
            sa.Integer,
            nullable=True,
            comment='调用耗时(毫秒)',
        ),
    )


def downgrade():
    """Downgrade: 回滚 sys_ai_billing 成本归因字段"""
    op.drop_column('sys_ai_billing', 'latency_ms')
    op.drop_column('sys_ai_billing', 'error_code')
    op.drop_column('sys_ai_billing', 'provider_id')
    op.drop_column('sys_ai_billing', 'request_id')
