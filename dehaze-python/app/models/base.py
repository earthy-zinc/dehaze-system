"""
SQLAlchemy 基础模型

提供自动填充功能：create_time、update_time、create_by、update_by
"""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, BigInteger
from sqlalchemy.orm import declarative_base
from sqlalchemy import event

from app.extensions import mysql

Base = declarative_base()


class BaseModel(mysql.Model):
    """
    基础模型类

    提供以下功能：
    1. 自动填充 create_time（创建时间）
    2. 自动填充 update_time（更新时间）
    3. 自动填充 create_by（创建人 ID）
    4. 自动填充 update_by（修改人 ID）

    使用方法：
        方式 1: 继承 BaseModel
        class User(BaseModel):
            __tablename__ = 'sys_user'
            id = Column(BigInteger, primary_key=True)
            username = Column(String(64))

        方式 2: 直接在模型类中导入需要的字段
        from app.models.base import BaseModel

        class User(mysql.Model):
            __tablename__ = 'sys_user'
            id = Column(BigInteger, primary_key=True)
            username = Column(String(64))
            # 复制 BaseModel 的字段
            create_time = BaseModel.create_time
            update_time = BaseModel.update_time
            create_by = BaseModel.create_by
            update_by = BaseModel.update_by

    注意:
        1. create_time 在插入时自动填充
        2. update_time 在插入和更新时自动填充
        3. create_by 和 update_by 需要通过上下文（Flask request）获取当前用户 ID
        4. 如果没有当前用户 ID，create_by 和 update_by 会被设置为 None
    """

    __abstract__ = True  # 这是一个抽象基类，不会创建表

    # 公共字段
    create_time = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment='创建时间'
    )
    update_time = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment='更新时间'
    )
    create_by = Column(
        BigInteger,
        comment='创建人 ID'
    )
    update_by = Column(
        BigInteger,
        comment='修改人 ID'
    )

@event.listens_for(BaseModel, 'before_insert')
def set_create_fields(mapper, connection, target):
    """
    插入前回调：自动填充 create_time、update_time、create_by、update_by

    Args:
        mapper: SQLAlchemy mapper
        connection: 数据库连接
        target: 模型实例
    """
    try:
        # 设置创建时间
        if hasattr(target, 'create_time') and target.create_time is None:
            target.create_time = datetime.now(timezone.utc)

        # 设置更新时间
        if hasattr(target, 'update_time'):
            target.update_time = datetime.now(timezone.utc)

        # 设置创建人 ID 和修改人 ID
        _set_user_fields(target)

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"插入前自动填充失败: {str(e)}", exc_info=True)


@event.listens_for(BaseModel, 'before_update')
def set_update_fields(mapper, connection, target):
    """
    更新前回调：自动填充 update_time 和 update_by

    Args:
        mapper: SQLAlchemy mapper
        connection: 数据库连接
        target: 模型实例
    """
    try:
        # 设置更新时间
        if hasattr(target, 'update_time'):
            target.update_time = datetime.now(timezone.utc)

        # 设置修改人 ID
        _set_user_fields(target, only_update=True)

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"更新前自动填充失败: {str(e)}", exc_info=True)


def _set_user_fields(target, only_update: bool = False):
    """
    设置用户字段（create_by 和 update_by）

    Args:
        target: 模型实例
        only_update: 是否只设置 update_by（不设置 create_by）
    """
    try:
        # 尝试获取当前用户 ID
        user_id = _get_current_user_id()

        if not only_update:
            # 设置创建人 ID
            if hasattr(target, 'create_by') and target.create_by is None:
                target.create_by = user_id

        # 设置修改人 ID
        if hasattr(target, 'update_by'):
            target.update_by = user_id

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"设置用户字段失败: {str(e)}")


def _get_current_user_id():
    """
    获取当前用户 ID

    从 Flask request 中获取当前用户 ID。

    Returns:
        int: 当前用户 ID，如果无法获取则返回 None
    """
    try:
        from flask import request, has_request_context

        if not has_request_context():
            return None

        # 尝试从 request.current_user_id 获取（由 JWT 装饰器设置）
        if hasattr(request, 'current_user_id'):
            return request.current_user_id

        # 尝试从 JWT token 中获取
        try:
            from flask_jwt_extended import get_jwt
            token = get_jwt()
            user_id = token.get('sub') or token.get('user_id')
            return user_id
        except Exception:
            pass

        return None

    except Exception:
        return None
