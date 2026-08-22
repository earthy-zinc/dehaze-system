"""
SQLAlchemy 基础模型

提供自动填充功能：create_time、update_time、create_by、update_by（BaseModel）；
AppendOnlyModel 仅自动填充 create_time（日志/流水/历史表基类）。
"""

from contextvars import ContextVar
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, event
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.core.constants import SYSTEM_USER_ID
from app.database import Base

_current_user_id: ContextVar[int | None] = ContextVar("current_user_id", default=None)


def set_current_user_id(user_id: int | None) -> None:
    _current_user_id.set(user_id)


def get_current_user_id() -> int | None:
    return _current_user_id.get()


class SoftDeleteMixin:
    """逻辑删除 Mixin。

    继承此 Mixin 的实体自动获得 deleted 列，
    并被全局 do_orm_execute 事件自动追加 deleted=0 过滤。
    """

    deleted: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment="逻辑删除标识(0:未删除;1:已删除)"
    )


class BaseModel(Base):
    """
    基础模型类

    提供以下功能：
    1. 自动填充 create_time（创建时间）
    2. 自动填充 update_time（更新时间）
    3. 自动填充 create_by（创建人 ID）
    4. 自动填充 update_by（修改人 ID）

    审计字段通过 Session.before_flush 事件统一填充，
    不使用 before_insert/before_update 事件（对 __abstract__ 基类不传播到子类）。
    """

    __abstract__ = True

    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        comment="创建时间",
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        comment="更新时间",
    )
    create_by: Mapped[int | None] = mapped_column(BigInteger, comment="创建人 ID")
    update_by: Mapped[int | None] = mapped_column(BigInteger, comment="修改人 ID")


class AppendOnlyModel(Base):
    """
    只追加模型类（日志/流水/历史表基类）

    不承载通用操作人审计（create_by/update_by），仅自动填充 create_time；
    操作人语义由业务字段表达（如 sys_ai_credit_log.operator_id、sys_ai_refund.auditor_id）。
    """

    __abstract__ = True

    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        comment="创建时间",
    )


@event.listens_for(Session, "before_flush")
def set_audit_fields(session, context, instances):
    """flush 前回调：填充审计字段

    BaseModel 填充 create_time/update_time/create_by/update_by；
    AppendOnlyModel（日志/流水表）仅填充 create_time。
    使用本地时间（与 Java LocalDateTime.now() / Go time.Now() 保持一致），
    避免 UTC 与本地时间混用导致按 create_time DESC 排序时新记录被排到旧记录之后。
    """
    now = datetime.now()
    for obj in session.new:
        if isinstance(obj, BaseModel):
            if obj.create_time is None:
                obj.create_time = now
            obj.update_time = now
            _set_user_fields(obj)
        elif isinstance(obj, AppendOnlyModel):
            if obj.create_time is None:
                obj.create_time = now
    for obj in session.dirty:
        if isinstance(obj, BaseModel):
            obj.update_time = now
            _set_user_fields(obj, only_update=True)


def _set_user_fields(target, only_update: bool = False):
    user_id = get_current_user_id()
    if user_id is None:
        user_id = SYSTEM_USER_ID

    if not only_update:
        if target.create_by is None:
            target.create_by = user_id

    target.update_by = user_id


def get_audit_update_values() -> dict:
    """
    获取审计字段更新值（用于 SQLAlchemy Core update 语句）

    Core update 绕过 ORM 事件，需手动填充 update_by 和 update_time。
    """
    user_id = get_current_user_id()
    if user_id is None:
        user_id = SYSTEM_USER_ID
    return {
        "update_by": user_id,
        "update_time": datetime.now(),
    }
