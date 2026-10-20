from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiAgentMcp(AppendOnlyModel):
    __tablename__ = "sys_ai_agent_mcp"
    __table_args__ = {"comment": "Agent-MCP命名空间关联表"}

    agent_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, comment="关联Agent ID(关联sys_ai_agent.id)"
    )
    mcp_namespace: Mapped[str] = mapped_column(
        String(64), primary_key=True, comment="MCP命名空间(如image_processing/evaluation)"
    )
