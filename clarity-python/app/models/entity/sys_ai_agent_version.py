from typing import Any

from sqlalchemy import BigInteger, Index, Integer, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiAgentVersion(AppendOnlyModel):
    __tablename__ = "sys_ai_agent_version"
    __table_args__ = (
        Index("uk_agent_version", "agent_id", "version_no", unique=True),
        Index("idx_agent", "agent_id"),
        {"comment": "AI智能体配置版本快照表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    agent_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联Agent ID(关联sys_ai_agent.id)"
    )
    version_no: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="版本号(每个Agent内自增,不可变、不回填)"
    )
    snapshot: Mapped[Any] = mapped_column(
        JSON,
        nullable=False,
        comment="配置快照JSON(系统提示词/模型/推理参数/Skills/MCP/子Agent关联/权限/护栏的完整序列化)",
    )
    status: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=1,
        comment="版本状态(1:草稿;2:已发布;同一Agent同一时刻至多一条已发布)",
    )
    change_note: Mapped[str | None] = mapped_column(String(512), nullable=True, comment="变更说明")
    operator_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment="操作人ID")
