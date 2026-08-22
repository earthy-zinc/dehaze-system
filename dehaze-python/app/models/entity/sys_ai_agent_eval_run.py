from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Index, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiAgentEvalRun(AppendOnlyModel):
    """智能体评测执行记录（只追加审计轨迹，不逻辑删除）。"""

    __tablename__ = "sys_ai_agent_eval_run"
    __table_args__ = (
        Index("idx_agent", "agent_id"),
        Index("idx_dataset", "dataset_id"),
        {"comment": "AI智能体评测执行记录表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    agent_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联Agent ID(关联sys_ai_agent.id)"
    )
    dataset_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联评测集ID(关联sys_ai_agent_eval_dataset.id)"
    )
    trigger_type: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="触发方式(manual:手动触发;publish:发布门禁内部触发)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="执行状态(1:执行中;2:通过;3:失败)"
    )
    score_summary: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="四维评分聚合JSON(结果质量/过程合规/安全边界/效率)"
    )
    results: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="样本明细JSON(每条样本的四维分数+通过状态+差异说明)"
    )
    create_by: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="创建人ID(触发评测的用户)"
    )
    update_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="更新时间"
    )
