from sqlalchemy import BigInteger, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysMemberBenefit(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_member_benefit"
    __table_args__ = {"comment": "会员等级权益配置表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    level_code: Mapped[str] = mapped_column(String(16), nullable=False, comment="会员等级")
    level_name: Mapped[str] = mapped_column(String(32), nullable=False, comment="等级名称")
    growth_min: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="成长值下限"
    )
    growth_max: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="成长值上限(0表示无上限)"
    )
    monthly_dehaze_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="月度去雾次数配额"
    )
    monthly_derain_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="月度去雨次数配额"
    )
    monthly_desnow_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="月度去雪次数配额"
    )
    monthly_lowlight_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="月度低光增强次数配额"
    )
    monthly_super_resolution_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="月度超分辨率次数配额"
    )
    monthly_denoise_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="月度去噪次数配额"
    )
    monthly_inpaint_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="月度图像修复次数配额"
    )
    monthly_evaluate_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="月度评估次数配额"
    )
    history_retention: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="历史记录保留条数"
    )
    batch_limit: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="批量处理上限(张)"
    )
    priority: Mapped[int] = mapped_column(
        mysql_types.TINYINT,
        nullable=False,
        default=1,
        comment="处理优先级(1:普通;2:优先;3:高优先;4:最高)",
    )
    advanced_params: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment="高级参数调节(0:关闭;1:开启)"
    )
    hd_export: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment="高清图导出(0:关闭;1:开启)"
    )
    report_export: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment="对比报告导出(0:关闭;1:开启)"
    )
    batch_download: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment="批量打包下载(0:关闭;1:开启)"
    )
    ai_credits_daily: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="AI对话日限额(积分/天，每日0点重置)"
    )
    ai_credits_monthly: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="AI对话月限额(积分/月，每月1日重置)"
    )
    vip_gift_credits: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="VIP按月赠送积分(0表示该等级不赠送)"
    )
    multimodal_limit: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="多模态视觉读取日限额(次/天，每日0点重置)"
    )
    sort: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="排序值")
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
