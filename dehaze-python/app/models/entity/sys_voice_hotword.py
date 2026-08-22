from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysVoiceHotword(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_voice_hotword"
    __table_args__ = {"comment": "语音热词表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    word: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="热词内容(XSS转义后存储)"
    )
    scope: Mapped[str] = mapped_column(
        String(16), nullable=False, default="user", comment="作用域(global:全局;user:用户级)"
    )
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, index=True, nullable=True, comment="所属用户ID(关联sys_user.id，global时为NULL)"
    )
