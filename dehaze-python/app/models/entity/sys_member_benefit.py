from app.models.base import BaseModel
from sqlalchemy import BigInteger, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column


class SysMemberBenefit(BaseModel):
    __tablename__ = 'sys_member_benefit'
    __table_args__ = {'comment': '会员等级权益配置表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    level_code: Mapped[str] = mapped_column(String(16), nullable=False, comment='会员等级')
    level_name: Mapped[str] = mapped_column(String(32), nullable=False, comment='等级名称')
    growth_min: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment='成长值下限')
    growth_max: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment='成长值上限(0表示无上限)')
    monthly_dehaze_quota: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='月度去雾次数配额')
    monthly_evaluate_quota: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='月度评估次数配额')
    history_retention: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='历史记录保留条数')
    batch_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='批量处理上限(张)')
    priority: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=1, comment='处理优先级(1:普通;2:优先;3:高优先;4:最高)')
    advanced_params: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='高级参数调节(0:关闭;1:开启)')
    hd_export: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='高清图导出(0:关闭;1:开启)')
    report_export: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='对比报告导出(0:关闭;1:开启)')
    batch_download: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='批量打包下载(0:关闭;1:开启)')
    sort: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='排序值')
    status: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=1, comment='状态(1:启用;0:禁用)')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识')
