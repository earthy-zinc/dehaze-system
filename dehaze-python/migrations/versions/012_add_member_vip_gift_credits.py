"""add vip_gift_credits to sys_member_benefit

Revision ID: 012_add_member_vip_gift_credits
Revises: 011_add_agent_thought_summary
Create Date: 2026-08-20

VIP 按月赠送积分：为 sys_member_benefit 新增 vip_gift_credits 字段，
承载各等级月度赠送积分的差异化配置（AI 计费 F-MB-002 §2.2.2），
由 grantVipMonthlyGift 定时任务每月 1 日发放，月末未用部分清零。
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '012_add_member_vip_gift_credits'
down_revision = '011_add_agent_thought_summary'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 为 sys_member_benefit 添加 VIP 月度赠送积分字段"""
    op.add_column(
        'sys_member_benefit',
        sa.Column(
            'vip_gift_credits',
            sa.BigInteger,
            nullable=False,
            server_default='0',
            comment='VIP按月赠送积分(0表示该等级不赠送)',
        ),
    )


def downgrade():
    """Downgrade: 回滚 sys_member_benefit 的 VIP 月度赠送积分字段"""
    op.drop_column('sys_member_benefit', 'vip_gift_credits')
