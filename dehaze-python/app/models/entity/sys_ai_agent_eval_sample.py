from typing import Any

from sqlalchemy import BigInteger, Index, String, Text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysAiAgentEvalSample(BaseModel):
    __tablename__ = "sys_ai_agent_eval_sample"
    __table_args__ = (Index("idx_dataset", "dataset_id"), {"comment": "AI智能体评测样本表"})

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    dataset_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联评测集ID(关联sys_ai_agent_eval_dataset.id)"
    )
    task_goal: Mapped[str] = mapped_column(
        Text, nullable=False, comment="任务目标(样本要完成的任务描述)"
    )
    allowed_input: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="允许输入(输入范围/约束说明,可空)"
    )
    tools: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="可用工具JSON(样本预期调用的工具列表,可空)"
    )
    expected_process: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="期望过程(正确的推理/调用过程,可空)"
    )
    expected_result: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="期望结果(准确/完整/格式等通过条件,可空)"
    )
    forbidden_behavior: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="禁止行为(不得发生的越权/注入/敏感泄露等,可空)"
    )
    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="low",
        comment="风险等级(low:低;medium:中;high:高,high样本失败阻断发布)",
    )
