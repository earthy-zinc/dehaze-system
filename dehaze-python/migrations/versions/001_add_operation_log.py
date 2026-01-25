"""add operation log table

Revision ID: add_operation_log
Revises: 
Create Date: 2026-01-17

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime, timezone


# revision identifiers, used by Alembic.
revision = 'add_operation_log'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 创建操作日志表"""
    op.create_table(
        'sys_operation_log',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False, comment='日志ID'),
        sa.Column('ip', sa.String(64), nullable=False, server_default='', comment='请求IP地址'),
        sa.Column('method', sa.String(10), nullable=False, comment='请求方法'),
        sa.Column('path', sa.String(255), nullable=False, comment='请求路径'),
        sa.Column('status', sa.Integer(), nullable=False, server_default='200', comment='响应状态码'),
        sa.Column('latency', sa.Integer(), nullable=False, server_default='0', comment='请求耗时(毫秒)'),
        sa.Column('agent', sa.String(512), server_default='', comment='用户代理'),
        sa.Column('error_message', sa.Text(), server_default='', comment='错误信息'),
        sa.Column('body', sa.Text(), server_default='', comment='请求体(JSON字符串)'),
        sa.Column('resp', sa.Text(), server_default='', comment='响应体(JSON字符串)'),
        sa.Column('user_id', sa.BigInteger(), nullable=True, comment='用户ID'),
        sa.Column('create_time', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.PrimaryKeyConstraint('id'),
        comment='操作日志表'
    )
    # 创建索引
    op.create_index('idx_user_id', 'sys_operation_log', ['user_id'])
    op.create_index('idx_create_time', 'sys_operation_log', ['create_time'])
    op.create_index('idx_status', 'sys_operation_log', ['status'])


def downgrade():
    """Downgrade: 删除操作日志表"""
    op.drop_index('idx_status', table_name='sys_operation_log')
    op.drop_index('idx_create_time', table_name='sys_operation_log')
    op.drop_index('idx_user_id', table_name='sys_operation_log')
    op.drop_table('sys_operation_log')
