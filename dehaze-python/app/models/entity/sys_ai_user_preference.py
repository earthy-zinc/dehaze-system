from sqlalchemy import BigInteger, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiUserPreference(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_user_preference"
    __table_args__ = (
        # 每用户一行（与软删标记联合唯一，支持软删后重建）
        UniqueConstraint("user_id", "deleted", name="uk_user_id"),
        {"comment": "用户AI偏好设置表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    default_model: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="偏好默认对话模型(model_id;空=未设置,回落平台默认模型)",
    )
