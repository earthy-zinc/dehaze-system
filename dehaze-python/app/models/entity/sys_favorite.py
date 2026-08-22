"""
统一收藏实体

对齐 Java SysFavorite.java + config/sql/schema/sys_favorite.sql
通过 target_type + target_id 实现多态收藏，支持 algorithm/result/dataset 等任意业务实体。
"""

from sqlalchemy import BigInteger, Index, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysFavorite(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_favorite"
    __table_args__ = (
        Index("uk_user_target", "user_id", "target_type", "target_id", unique=True),
        Index("idx_user_type_time", "user_id", "target_type", "create_time"),
        {"comment": "统一收藏表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    target_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="收藏对象类型(algorithm:算法;result:处理结果;dataset:数据集;image:图片;preset:参数方案)",
    )
    target_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="收藏对象ID")
    is_invalid: Mapped[int] = mapped_column(
        mysql_types.TINYINT,
        nullable=False,
        default=0,
        comment="收藏对象是否已失效(0:正常;1:已失效)",
    )
