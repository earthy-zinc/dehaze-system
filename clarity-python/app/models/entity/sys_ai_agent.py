from typing import Any

from sqlalchemy import BigInteger, Index, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiAgent(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_agent"
    __table_args__ = (
        Index("uk_agent_code", "agent_code", unique=True),
        Index("idx_model", "model_id"),
        Index("idx_status_type", "status", "is_subagent", "is_team"),
        {"comment": "AI智能体配置表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    agent_code: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="Agent唯一编码(业务引用键,如default;image-analyst)"
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="Agent显示名称")
    description: Mapped[str] = mapped_column(
        String(512), nullable=False, default="", comment="Agent描述(供LLM决策调用时参考)"
    )
    system_prompt: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="系统提示词(Markdown,为空时由deepagents使用内置默认)"
    )
    model_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="关联模型标识(关联sys_ai_model.model_id)"
    )
    reasoning_mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="auto",
        comment="推理范式(auto:复杂度评估自动选择;direct:直接回复;react:边想边做;plan_execute:先想后做;reflexion:反思迭代)",
    )
    config: Mapped[Any | None] = mapped_column(
        JSON,
        nullable=True,
        comment="推理参数JSON(max_steps/token_budget/max_parallel/tool_timeout/retry_max/reflexion_threshold/temperature等,为空继承sys_dict系统默认)",
    )
    is_subagent: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=0,
        comment="是否可作为子Agent(0:否;1:是,不可被会话直接选择)",
    )
    is_team: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=0,
        comment="是否为Team团队(0:否;1:是,通过langgraph-supervisor编排多Agent协作)",
    )
    is_exposed: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=0,
        comment="是否对外暴露为A2A子Agent(0:否;1:是,默认不暴露;仅启用且非子Agent的普通Agent可暴露)",
    )
    permissions: Mapped[Any | None] = mapped_column(
        JSON,
        nullable=True,
        comment=(
            "文件系统权限规则JSON(deepagents FilesystemPermission:operations/paths/mode,"
            "mode支持allow/deny/interrupt)"
        ),
    )
    tags: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="分类标签(字符串数组,管理端筛选/展示用)"
    )
    sort_order: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="排序序号(数字越小越靠前)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
