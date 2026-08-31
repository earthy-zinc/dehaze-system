from sqlalchemy import BigInteger, Index, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysAiEvalReview(BaseModel):
    """智能体评测人工复核记录（判分结果抽样 + 人工判定回填）。"""

    __tablename__ = "sys_ai_eval_review"
    __table_args__ = (
        Index("uk_run_sample", "run_id", "sample_id", unique=True),
        Index("idx_agent_status", "agent_id", "status"),
        {"comment": "AI智能体评测人工复核表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    run_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联评测执行ID(关联sys_ai_agent_eval_run.id)"
    )
    sample_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联评测样本ID(关联sys_ai_agent_eval_sample.id)"
    )
    agent_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="关联Agent ID(冗余自sys_ai_agent_eval_run.agent_id,按Agent聚合查询)",
    )
    judge_passed: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, comment="判分模型判定(1:通过;0:失败)"
    )
    risk_level: Mapped[str] = mapped_column(
        String(10), nullable=False, default="low", comment="样本风险等级快照(low/medium/high)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="复核状态(1:待复核;2:已复核)"
    )
    agree: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True, comment="人工判定(1:与判分一致;0:不一致)"
    )
    reviewer_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="复核人ID(关联sys_user.id)"
    )
    remark: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment="复核备注"
    )
