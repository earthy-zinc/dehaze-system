from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SysMemberSignIn(Base):
    __tablename__ = 'sys_member_sign_in'
    __table_args__ = {'comment': '会员签到记录表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    sign_date: Mapped[date] = mapped_column(Date, nullable=False, comment='签到日期')
    continuous_days: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment='连续签到天数')
    growth_value: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='本次获得成长值')
    create_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now, comment='创建时间')
