"""add status/error_message columns to sys_pred_log and sys_eval_log

Revision ID: add_pred_eval_log_status
Revises: add_algo_version_audit_img_history
Create Date: 2026-07-25

为预测/评估日志表新增任务状态字段，支持异步任务模式：
- sys_pred_log.status: processing/completed/failed（默认 completed，兼容历史数据）
- sys_pred_log.error_message: 失败错误信息
- sys_eval_log.status: 同上
- sys_eval_log.error_message: 同上
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_pred_eval_log_status'
down_revision = 'add_algo_version_audit_img_history'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 为 sys_pred_log / sys_eval_log 添加 status 和 error_message 字段"""
    op.add_column(
        'sys_pred_log',
        sa.Column(
            'status',
            sa.String(20),
            nullable=False,
            server_default='completed',
            comment='任务状态：processing/completed/failed',
        ),
    )
    op.add_column(
        'sys_pred_log',
        sa.Column(
            'error_message',
            sa.Text,
            nullable=True,
            comment='失败错误信息',
        ),
    )
    op.create_index('idx_status', 'sys_pred_log', ['status'])

    op.add_column(
        'sys_eval_log',
        sa.Column(
            'status',
            sa.String(20),
            nullable=False,
            server_default='completed',
            comment='任务状态：processing/completed/failed',
        ),
    )
    op.add_column(
        'sys_eval_log',
        sa.Column(
            'error_message',
            sa.Text,
            nullable=True,
            comment='失败错误信息',
        ),
    )
    op.create_index('idx_status', 'sys_eval_log', ['status'])


def downgrade():
    """Downgrade: 回滚 sys_pred_log / sys_eval_log 状态字段"""
    op.drop_index('idx_status', table_name='sys_eval_log')
    op.drop_column('sys_eval_log', 'error_message')
    op.drop_column('sys_eval_log', 'status')

    op.drop_index('idx_status', table_name='sys_pred_log')
    op.drop_column('sys_pred_log', 'error_message')
    op.drop_column('sys_pred_log', 'status')
