"""add algorithm version/audit fields, input history, algorithm version tables

Revision ID: add_algo_version_audit_img_history
Revises: add_file_size_bytes
Create Date: 2026-07-11

对齐 dehaze-java 权威 schema:
1. sys_algorithm 表新增字段: version, audit_by, audit_time, audit_remark
2. 新表 sys_algorithm_version: 算法版本历史表
3. 新表 sys_input_history: 图像输入历史记录表
4. 新表 sys_algorithm_favorite: 算法收藏表 (Python 独有功能, Java/Go 不访问)

注: 本迁移脚本 1-3 项完全对齐 dehaze-java/config/sql/schema.sql 中已定义的结构,
   三端(dehaze-java/dehaze-go/dehaze-python)共享同一数据库 schema.
   第 4 项 sys_algorithm_favorite 为 Python 端独有功能表, 不影响 Java/Go.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision = 'add_algo_version_audit_img_history'
down_revision = 'add_file_size_bytes'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade: 新增算法版本/审核字段 + 算法版本历史表 + 图像输入历史表 + 算法收藏表"""

    # 1. sys_algorithm 表新增字段 (对齐 Java schema.sql)
    op.add_column(
        'sys_algorithm',
        sa.Column('version', sa.String(50), nullable=True, comment='算法版本号'),
    )
    op.add_column(
        'sys_algorithm',
        sa.Column('audit_by', sa.BigInteger, nullable=True, comment='审核人ID'),
    )
    op.add_column(
        'sys_algorithm',
        sa.Column('audit_time', sa.DateTime, nullable=True, comment='审核时间'),
    )
    op.add_column(
        'sys_algorithm',
        sa.Column('audit_remark', sa.String(500), nullable=True, comment='审核备注'),
    )

    # 2. 新表 sys_algorithm_version: 算法版本历史表 (对齐 Java schema.sql)
    op.create_table(
        'sys_algorithm_version',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True, comment='主键'),
        sa.Column('algorithm_id', sa.BigInteger, nullable=False, comment='关联算法ID'),
        sa.Column('version', sa.String(50), nullable=False, comment='版本号'),
        sa.Column('change_log', sa.Text, nullable=True, comment='变更日志'),
        sa.Column('status', sa.Integer, nullable=True, comment='该版本时的状态'),
        sa.Column('config_json', sa.Text, nullable=True, comment='该版本时的配置JSON'),
        sa.Column('model_file_id', sa.BigInteger, nullable=True, comment='模型文件ID'),
        sa.Column('is_active', mysql.TINYINT(1), nullable=True, server_default='0', comment='是否当前活跃版本'),
        sa.Column('create_time', sa.DateTime, nullable=False, server_default=sa.func.now(), comment='创建时间'),
        sa.Column('update_time', sa.DateTime, nullable=False, server_default=sa.func.now(), onupdate=sa.func.now(), comment='更新时间'),
        sa.Column('create_by', sa.BigInteger, nullable=True, comment='创建人ID'),
        sa.Column('update_by', sa.BigInteger, nullable=True, comment='修改人ID'),
        comment='算法版本历史表',
    )
    op.create_unique_constraint('uk_algo_version', 'sys_algorithm_version', ['algorithm_id', 'version'])
    op.create_index('idx_algorithm_id', 'sys_algorithm_version', ['algorithm_id'])

    # 3. 新表 sys_input_history: 图像输入历史记录表 (对齐 Java schema.sql)
    op.create_table(
        'sys_input_history',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True, comment='主键'),
        sa.Column('user_id', sa.BigInteger, nullable=False, comment='用户ID'),
        sa.Column('original_image_url', sa.String(500), nullable=True, comment='原始图片URL'),
        sa.Column('original_thumbnail_url', sa.String(500), nullable=True, comment='原始缩略图URL'),
        sa.Column('result_image_url', sa.String(500), nullable=True, comment='处理结果图片URL'),
        sa.Column('result_thumbnail_url', sa.String(500), nullable=True, comment='结果缩略图URL'),
        sa.Column('algorithm_id', sa.BigInteger, nullable=True, comment='算法ID'),
        sa.Column('algorithm_name', sa.String(100), nullable=True, comment='算法名称（冗余）'),
        sa.Column('algorithm_params', sa.Text, nullable=True, comment='算法参数（JSON）'),
        sa.Column('processing_time', sa.Integer, nullable=True, comment='处理耗时（毫秒）'),
        sa.Column('status', mysql.TINYINT, nullable=True, server_default='3', comment='处理状态（1=成功，2=失败，3=处理中）'),
        sa.Column('input_source', sa.String(20), nullable=True, comment='图片来源（upload/camera/sample）'),
        sa.Column('is_favorite', mysql.TINYINT(1), nullable=True, server_default='0', comment='是否收藏'),
        sa.Column('sync_status', mysql.TINYINT, nullable=True, server_default='0', comment='同步状态（0=未同步，1=已同步）'),
        sa.Column('create_time', sa.DateTime, nullable=False, server_default=sa.func.now(), comment='创建时间'),
        sa.Column('update_time', sa.DateTime, nullable=False, server_default=sa.func.now(), onupdate=sa.func.now(), comment='更新时间'),
        sa.Column('create_by', sa.BigInteger, nullable=True, comment='创建人ID'),
        sa.Column('update_by', sa.BigInteger, nullable=True, comment='修改人ID'),
        comment='图像输入历史记录表',
    )
    op.create_index('idx_user_time', 'sys_input_history', ['user_id', sa.text('create_time DESC')])
    op.create_index('idx_user_favorite', 'sys_input_history', ['user_id', 'is_favorite', sa.text('create_time DESC')])

    # 4. 新表 sys_algorithm_favorite: 算法收藏表 (Python 独有功能)
    op.create_table(
        'sys_algorithm_favorite',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True, comment='id'),
        sa.Column('user_id', sa.BigInteger, nullable=False, comment='用户ID'),
        sa.Column('algorithm_id', sa.BigInteger, nullable=False, comment='算法ID'),
        sa.Column('create_time', sa.DateTime, nullable=False, server_default=sa.func.now(), comment='收藏时间'),
        comment='算法收藏表',
    )
    op.create_index('idx_algo_favorite_user_id', 'sys_algorithm_favorite', ['user_id'])
    op.create_index('idx_algo_favorite_algorithm_id', 'sys_algorithm_favorite', ['algorithm_id'])
    op.create_unique_constraint('uk_user_algorithm', 'sys_algorithm_favorite', ['user_id', 'algorithm_id'])


def downgrade():
    """Downgrade: 回滚所有变更"""
    # 删除算法收藏表 (Python 独有)
    op.drop_table('sys_algorithm_favorite')

    # 删除图像输入历史表
    op.drop_table('sys_input_history')

    # 删除算法版本历史表
    op.drop_table('sys_algorithm_version')

    # 回滚 sys_algorithm 字段
    op.drop_column('sys_algorithm', 'audit_remark')
    op.drop_column('sys_algorithm', 'audit_time')
    op.drop_column('sys_algorithm', 'audit_by')
    op.drop_column('sys_algorithm', 'version')
