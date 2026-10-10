"""
日志相关实体模型
"""

from typing import Any

from sqlalchemy import CHAR, JSON, BigInteger, Index, Integer, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel
from app.models.enum.log_status import LogStatus


class SysPredLog(BaseModel):
    __tablename__ = "sys_pred_log"
    __table_args__ = (
        Index("idx_model_id", "model_id"),
        Index("idx_origin_md5", "origin_md5"),
        Index("idx_pred_md5", "pred_md5"),
        Index("idx_status", "status"),
        {"comment": "图像任务日志表"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="id")
    model_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="图像模型标识（统一模型规范 slug）"
    )
    provider_id: Mapped[int | None] = mapped_column(
        BigInteger, comment="实际调用供应商id（归因，降级后为备用供应商）"
    )
    actual_model: Mapped[str | None] = mapped_column(
        String(64), comment="实际执行模型（降级/路由对齐 chat 链路）"
    )
    vendor_task_id: Mapped[str | None] = mapped_column(
        String(128), comment="异步厂商任务号（dashscope 等，本地同步推理为空）"
    )
    origin_file_id: Mapped[int | None] = mapped_column(
        BigInteger, comment="原始图像文件id（有雾图像）"
    )
    origin_md5: Mapped[str] = mapped_column(CHAR(32), nullable=False, comment="原始图像md5值")
    origin_url: Mapped[str] = mapped_column(Text, nullable=False, comment="原始图像url")
    pred_file_id: Mapped[int | None] = mapped_column(BigInteger, comment="预测图像文件id")
    pred_md5: Mapped[str] = mapped_column(CHAR(32), nullable=False, comment="预测图像md5值")
    pred_url: Mapped[str] = mapped_column(Text, nullable=False, comment="预测图像url")
    recommended_by: Mapped[int | None] = mapped_column(
        BigInteger, comment="推荐来源：推荐记录ID（推荐管理模块，用于追踪推荐采纳率）"
    )
    time: Mapped[int] = mapped_column(Integer, default=0, comment="推理时间（秒）")
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT,
        nullable=False,
        default=LogStatus.COMPLETED.value,
        comment="任务状态(1:处理中;2:已完成;3:失败;4:已取消)",
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="失败错误信息")


class SysEvalLog(BaseModel):
    __tablename__ = "sys_eval_log"
    __table_args__ = (
        Index("idx_model_id", "model_id"),
        Index("idx_pred_md5", "pred_md5"),
        Index("idx_gt_md5", "gt_md5"),
        Index("idx_status", "status"),
        {"comment": "模型评估日志表"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="id")
    model_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="图像模型标识（统一模型规范 slug）"
    )
    pred_file_id: Mapped[int | None] = mapped_column(BigInteger, comment="预测图像文件id")
    pred_md5: Mapped[str] = mapped_column(CHAR(32), nullable=False, comment="预测图像md5值")
    pred_url: Mapped[str] = mapped_column(Text, nullable=False, comment="预测图像url")
    gt_file_id: Mapped[int | None] = mapped_column(BigInteger, comment="真值图像文件id")
    gt_md5: Mapped[str] = mapped_column(CHAR(32), nullable=False, comment="真值图像md5值")
    gt_url: Mapped[str] = mapped_column(Text, nullable=False, comment="真值图像url")
    time: Mapped[int] = mapped_column(Integer, default=0, comment="评估时间（秒）")
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT,
        nullable=False,
        default=LogStatus.COMPLETED.value,
        comment="任务状态(1:处理中;2:已完成;3:失败)",
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="失败错误信息")
    task_type: Mapped[str] = mapped_column(
        mysql_types.VARCHAR(20),
        nullable=False,
        default="evaluation",
        comment="任务类型(evaluation:效果评估;report:对比报告)",
    )
    result: Mapped[Any | None] = mapped_column(JSON, nullable=True, comment="预测结果")
