"""AI 对话 Skill 主表实体（F-M08-006 Skills 管理）。

对应表 sys_ai_skill：承载 Skill 元数据与 Markdown 指令全文。
skill_name 与 sys_ai_agent_skill.skill_name 保持外键语义（不加物理外键，对齐项目惯例）；
status 标识启停（0=禁用，1=启用，对齐 SDK SkillVO.status），禁用后不出现在 SkillManager 索引中。
配置类表使用逻辑删除（SoftDeleteMixin），删除前须校验被 Agent 关联。
"""

from sqlalchemy import BigInteger, Index, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiSkill(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_skill"
    __table_args__ = (
        Index("uk_name", "name", unique=True),
        Index("idx_status", "status"),
        {"comment": "AI对话Skill主表(F-M08-006)"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    name: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="Skill名称(唯一,关联sys_ai_agent_skill.skill_name)"
    )
    description: Mapped[str] = mapped_column(
        String(500), nullable=False, comment="Skill描述(供LLM渐进式加载索引使用)"
    )
    instruction: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Markdown指令全文(skill_load时完整注入)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="启停状态(0:禁用;1:启用)"
    )
    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="admin",
        comment="来源(builtin:内置播种;admin:管理员创建)",
    )
    market_shared: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=0,
        comment="是否共享至Skill市场(0:否;1:是)",
    )
