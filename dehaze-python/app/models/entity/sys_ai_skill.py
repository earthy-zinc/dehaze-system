"""AI 对话 Skill 主表实体（F-M08-006 Skills 管理）。

对应表 sys_ai_skill：承载 Skill 元数据与 SKILL.md 指令正文（遵循业界 Agent Skills 规范）。
skill_name 与 sys_ai_agent_skill.skill_name 保持外键语义（不加物理外键，对齐项目惯例）；
status 标识启停（0=禁用，1=启用，对齐 SDK SkillVO.status），禁用后不出现在 SkillManager 索引中。
license/compatibility/metadata/allowed_tools 对应 SKILL.md frontmatter 可选字段。
SKILL 目录内的其余文件（reference/ script/ assets/ README.md）存 sys_ai_skill_file 子表。
配置类表使用逻辑删除（SoftDeleteMixin），删除前须校验被 Agent 关联。
"""

from sqlalchemy import BigInteger, Index, JSON, SmallInteger, String, Text
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
    scene: Mapped[str] = mapped_column(
        String(255), nullable=False, default="", comment="适用场景(前端筛选/展示)"
    )
    instruction: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="SKILL.md指令正文(skill_load时注入,frontmatter之外的内容)"
    )
    license: Mapped[str | None] = mapped_column(
        String(255), nullable=True, comment="SKILL.md frontmatter license(许可证)"
    )
    compatibility: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment="SKILL.md frontmatter compatibility(环境要求)"
    )
    # 列名 metadata（frontmatter 键），Python 属性名 skill_metadata 避开 Declarative 保留字
    skill_metadata: Mapped[dict | None] = mapped_column(
        "metadata", JSON, nullable=True, comment="SKILL.md frontmatter metadata(任意键值,如版本/作者)"
    )
    allowed_tools: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="SKILL.md frontmatter allowed-tools(预批准工具列表,空格分隔)",
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
