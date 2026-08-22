from sqlalchemy import BigInteger, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiAgentSubagent(AppendOnlyModel):
    __tablename__ = "sys_ai_agent_subagent"
    __table_args__ = (
        Index("idx_subagent", "subagent_agent_id"),
        {"comment": "Agent-Subagent关联表"},
    )

    parent_agent_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, comment="父Agent ID(关联sys_ai_agent.id,主Agent或Team Lead)"
    )
    subagent_agent_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        comment=(
            "子Agent ID(关联sys_ai_agent.id,被调用的子Agent/Team Member;"
            "远程A2A子Agent为本地影子记录)"
        ),
    )
    endpoint_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="外部A2A端点ID(关联sys_ai_agent_endpoint.id;NULL为本地子Agent走task工具,非NULL为远程A2A子Agent走A2A客户端)",
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="优先级(数字越小越优先,多个子Agent均可处理同一任务时按此排序)",
    )
