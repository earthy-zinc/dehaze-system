"""AI 对话 Skill 目录文件清单子表实体（F-M08-006 Skills 管理）。

对应表 sys_ai_skill_file：承载 SKILL 目录内除 SKILL.md 外的文件清单
（reference/ script/ assets/ README.md 等，遵循业界 Agent Skills 规范）。
path 为相对 SKILL 根目录的路径，唯一约束 (skill_id, path)。
文件内容存对象存储（MinIO，对象 key = skills/{name}/{path}），本表只存清单
（path/file_size/file_type），支持渐进披露第三级（资源按需加载）。
"""

from sqlalchemy import BigInteger, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysAiSkillFile(BaseModel):
    __tablename__ = "sys_ai_skill_file"
    __table_args__ = (
        Index("uk_skill_path", "skill_id", "path", unique=True),
        {"comment": "AI对话Skill目录文件清单表(F-M08-006)"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    skill_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="所属Skill主键(sys_ai_skill.id)"
    )
    path: Mapped[str] = mapped_column(
        String(500), nullable=False, comment="相对SKILL根目录的文件路径"
    )
    file_size: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="文件大小(字节)"
    )
    file_type: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="文件类型(MIME或扩展名)"
    )
