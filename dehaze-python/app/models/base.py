"""
SQLAlchemy 基础模型

提供自动填充功能：create_time、update_time、create_by、update_by
"""

from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

from app.database import Base
from sqlalchemy import BigInteger, DateTime, event
from sqlalchemy.orm import Mapped, mapped_column

# 用户上下文：存储当前请求的用户 ID
_current_user_id: ContextVar[Optional[int]] = ContextVar(
    "current_user_id", default=None)


def set_current_user_id(user_id: Optional[int]) -> None:
    _current_user_id.set(user_id)


def get_current_user_id() -> Optional[int]:
    """
    获取当前用户 ID

    Returns:
        当前用户 ID，未设置时返回 None
    """
    return _current_user_id.get()


class BaseModel(Base):
    """
    基础模型类

    提供以下功能：
    1. 自动填充 create_time（创建时间）
    2. 自动填充 update_time（更新时间）
    3. 自动填充 create_by（创建人 ID）
    4. 自动填充 update_by（修改人 ID）

    使用方法：
        class User(BaseModel):
            __tablename__ = 'sys_user'
            __abstract__ = True
            id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
            username: Mapped[str] = mapped_column(String(64))
    """

    __abstract__ = True  # 这是一个抽象基类，不会创建表

    # 公共字段
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="创建时间",
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="更新时间",
    )
    create_by: Mapped[int | None] = mapped_column(BigInteger, comment="创建人 ID")
    update_by: Mapped[int | None] = mapped_column(BigInteger, comment="修改人 ID")


@event.listens_for(BaseModel, "before_insert")
def set_create_fields(mapper, connection, target):
    """
    插入前回调：自动填充 create_time、update_time、create_by、update_by
    """
    try:
        if hasattr(target, "create_time") and target.create_time is None:
            target.create_time = datetime.now(timezone.utc)

        if hasattr(target, "update_time"):
            target.update_time = datetime.now(timezone.utc)

        _set_user_fields(target)

    except Exception as e:
        import logging

        logging.getLogger(__name__).error(
            f"插入前自动填充失败: {str(e)}", exc_info=True)


@event.listens_for(BaseModel, "before_update")
def set_update_fields(mapper, connection, target):
    """
    更新前回调：自动填充 update_time 和 update_by
    """
    try:
        if hasattr(target, "update_time"):
            target.update_time = datetime.now(timezone.utc)

        _set_user_fields(target, only_update=True)

    except Exception as e:
        import logging

        logging.getLogger(__name__).error(
            f"更新前自动填充失败: {str(e)}", exc_info=True)


def _set_user_fields(target, only_update: bool = False):
    """
    设置用户字段（create_by 和 update_by）

    注意：FastAPI 中需要通过其他方式传递用户上下文（如 Depends）
    此处暂时保留 None，后续可通过 contextvars 实现
    """
    user_id = get_current_user_id()

    if not only_update:
        if hasattr(target, "create_by") and target.create_by is None:
            target.create_by = user_id

    if hasattr(target, "update_by"):
        target.update_by = user_id
