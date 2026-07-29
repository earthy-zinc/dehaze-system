from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SysMemberQuota(Base):
    __tablename__ = 'sys_member_quota'
    __table_args__ = {'comment': '会员月度配额历史表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    quota_month: Mapped[int] = mapped_column(Integer, nullable=False, comment='配额月份(yyyyMM)')
    level_code: Mapped[str] = mapped_column(String(16), nullable=False, comment='当月会员等级')
    dehaze_quota: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='当月去雾配额')
    dehaze_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='当月已用去雾次数')
    evaluate_quota: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='当月评估配额')
    evaluate_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='当月已用评估次数')
    reset_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment='配额重置时间')
    deleted: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0, comment='逻辑删除标识(0:未删除;1:已删除)')
    create_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment='创建时间')
    update_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')
    create_by: Mapped[int] = mapped_column(BigInteger, nullable=True, comment='创建人ID')
    update_by: Mapped[int] = mapped_column(BigInteger, nullable=True, comment='修改人ID')
