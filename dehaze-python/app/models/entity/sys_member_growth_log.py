from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SysMemberGrowthLog(Base):
    __tablename__ = 'sys_member_growth_log'
    __table_args__ = {'comment': '成长值流水表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    change_type: Mapped[str] = mapped_column(String(32), nullable=False, comment='变动类型')
    change_value: Mapped[int] = mapped_column(Integer, nullable=False, comment='变动值(正数增加/负数扣减)')
    balance: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='变动后成长值余额')
    related_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment='关联业务ID')
    reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, comment='变动原因')
    operator_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='操作人ID')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识(0:未删除;1:已删除)')
    create_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment='创建时间')
    update_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')
    create_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='创建人ID')
    update_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='修改人ID')
