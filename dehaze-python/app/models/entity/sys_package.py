from typing import Any, Optional

from sqlalchemy import BigInteger, Integer, String, JSON
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysPackage(BaseModel):
    __tablename__ = 'sys_package'
    __table_args__ = {'comment': '套餐表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    name: Mapped[str] = mapped_column(String(32), nullable=False, comment='套餐名称')
    level_code: Mapped[str] = mapped_column(String(16), nullable=False, comment='关联会员等级')
    period: Mapped[str] = mapped_column(String(16), nullable=False, comment='计费周期')
    period_days: Mapped[int] = mapped_column(Integer, nullable=False, comment='有效期天数')
    original_price: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='原价(分)')
    sale_price: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='促销价(分)')
    description: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, comment='套餐描述')
    benefit_overrides: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment='套餐权益覆盖项')
    sales_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment='销量')
    sort: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='排序值')
    status: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='上下架状态(1:上架;0:下架)')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识')
