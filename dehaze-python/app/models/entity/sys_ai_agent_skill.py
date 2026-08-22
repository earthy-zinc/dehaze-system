from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiAgentSkill(AppendOnlyModel):
    __tablename__ = "sys_ai_agent_skill"
    __table_args__ = {"comment": "Agent-Skill关联表"}

    agent_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, comment="关联Agent ID(关联sys_ai_agent.id)"
    )
    skill_name: Mapped[str] = mapped_column(
        String(128), primary_key=True, comment="Skill名称(关联sys_ai_skill.name)"
    )
