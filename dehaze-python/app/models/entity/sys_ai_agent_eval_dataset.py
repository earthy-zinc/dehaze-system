from sqlalchemy import BigInteger, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiAgentEvalDataset(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_agent_eval_dataset"
    __table_args__ = (
        Index("uk_agent_dataset_type", "agent_id", "dataset_type", unique=True),
        Index("idx_agent", "agent_id"),
        {"comment": "AI智能体评测集表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    agent_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联Agent ID(关联sys_ai_agent.id)"
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="评测集名称")
    description: Mapped[str] = mapped_column(
        String(512), nullable=False, default="", comment="评测集描述"
    )
    dataset_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="评测集类型(dev:开发集;regression:回归集;heldout:保留集)",
    )
