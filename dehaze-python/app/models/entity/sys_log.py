"""
日志相关实体模型
"""

from typing import Any

from sqlalchemy import CHAR, JSON, BigInteger, Index, Integer, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel
from app.models.enum.log_status import LogStatus


class SysPredLog(BaseModel):
    __tablename__ = "sys_pred_log"
    __table_args__ = (
        Index("idx_algorithm_id", "algorithm_id"),
        Index("idx_origin_md5", "origin_md5"),
        Index("idx_pred_md5", "pred_md5"),
        Index("idx_status", "status"),
        {"comment": "模型预测日志表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="id"
    )
    algorithm_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="算法id")
    origin_file_id: Mapped[int | None] = mapped_column(
        BigInteger, comment="原始图像文件id（有雾图像）"
    )
    origin_md5: Mapped[str] = mapped_column(CHAR(32), nullable=False, comment="原始图像md5值")
    origin_url: Mapped[str] = mapped_column(Text, nullable=False, comment="原始图像url")
    pred_file_id: Mapped[int | None] = mapped_column(BigInteger, comment="预测图像文件id")
    pred_md5: Mapped[str] = mapped_column(CHAR(32), nullable=False, comment="预测图像md5值")
    pred_url: Mapped[str] = mapped_column(Text, nullable=False, comment="预测图像url")
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
        Index("idx_algorithm_id", "algorithm_id"),
        Index("idx_pred_md5", "pred_md5"),
        Index("idx_gt_md5", "gt_md5"),
        Index("idx_status", "status"),
        {"comment": "模型预测日志表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="id"
    )
    algorithm_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="算法id")
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
    result: Mapped[Any | None] = mapped_column(JSON, nullable=True, comment="预测结果")
