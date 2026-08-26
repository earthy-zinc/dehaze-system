from sqlalchemy import JSON, BigInteger, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysPackage(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_package"
    __table_args__ = {"comment": "套餐表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    name: Mapped[str] = mapped_column(String(32), nullable=False, comment="套餐名称")
    package_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="vip",
        comment="商品类型(vip:会员卡;credit:积分卡;创建后不可修改)"
    )
    level_code: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="关联会员等级(level_1/level_2/level_3;积分卡为NULL)"
    )
    period: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="计费周期(monthly:月;quarterly:季;yearly:年;积分卡为NULL)"
    )
    period_days: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="有效期天数(积分卡为NULL)"
    )
    credit_amount: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="可得积分(积分卡商品;会员卡为NULL)"
    )
    original_price: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="原价(分)")
    sale_price: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="促销价(分)")
    description: Mapped[str | None] = mapped_column(String(256), nullable=True, comment="套餐描述")
    benefit_overrides: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="套餐权益覆盖项"
    )
    sales_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment="销量")
    sort: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="排序值")
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment="上下架状态(1:上架;0:下架)"
    )
