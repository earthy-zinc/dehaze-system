"""Skills 管理 Schema 模型（F-M08-006）。

请求/响应字段采用 snake_case 定义、序列化输出 camelCase（继承 OrmResult），
与 API 契约（API接口.md Skills 管理接口）对齐。
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from app.models.schema.common import BasePageQuery, OrmResult, validate_no_xss

# 指令内容上限（100KB）
CONTENT_MAX_BYTES = 100 * 1024

# 危险操作正则（命中即拦截，防止 Skill 指令被注入破坏性 shell 命令）
DANGEROUS_PATTERN = (
    r"(rm\s+-rf\s*/|mkfs\.?(ext\d?|xfs|vfat|ntfs)?\b|curl[^\n]*\|\s*(ba)?sh\b"
    r"|wget[^\n]*\|\s*(ba)?sh\b|sudo\s+(rm|shutdown|reboot|mkfs|dd)|dd\s+if=.*of=/dev/)"
)


class SkillCreate(OrmResult):
    """创建 Skill 请求体（SDK SkillForm：字段 instruction，另有 scene/scriptContent/templateId 可忽略）"""

    name: str = Field(..., min_length=1, max_length=128, description="Skill名称(唯一)")
    description: str = Field(..., min_length=1, max_length=500, description="Skill描述")
    scene: str = Field(default="", max_length=255, description="适用场景")
    instruction: str = Field(..., min_length=1, description="Markdown指令全文")

    @field_validator("name")
    @classmethod
    def _check_name(cls, v):
        return validate_no_xss(v.strip())

    @field_validator("description")
    @classmethod
    def _check_description(cls, v):
        return validate_no_xss(v.strip())


class SkillUpdate(OrmResult):
    """更新 Skill 请求体"""

    name: str | None = Field(default=None, min_length=1, max_length=128, description="Skill名称")
    description: str | None = Field(
        default=None, min_length=1, max_length=500, description="Skill描述"
    )
    scene: str | None = Field(default=None, max_length=255, description="适用场景")
    instruction: str | None = Field(default=None, min_length=1, description="Markdown指令全文")

    @field_validator("name")
    @classmethod
    def _check_name(cls, v):
        if v is None:
            return v
        return validate_no_xss(v.strip())

    @field_validator("description")
    @classmethod
    def _check_description(cls, v):
        if v is None:
            return v
        return validate_no_xss(v.strip())


class SkillStatusForm(OrmResult):
    """启停 Skill 请求体（SDK switchSkillStatus 传 {status: 0|1}）"""

    status: Literal[0, 1] = Field(..., description="目标启停状态(0:禁用;1:启用)")


class SkillFileVO(OrmResult):
    """SKILL 目录内资源文件清单项（内容存对象存储，仅返回清单供展示/按需加载）"""

    path: str = Field(description="相对SKILL根目录的文件路径")
    fileSize: int = Field(default=0, description="文件大小(字节)")
    fileType: str | None = Field(default=None, description="文件类型(MIME/扩展名)")


class SkillResult(OrmResult):
    """Skill 详情（管理员全部字段，instruction 映射实体 instruction）"""

    id: int = Field(description="主键")
    name: str = Field(description="Skill名称")
    description: str = Field(description="Skill描述")
    scene: str = Field(default="", description="适用场景")
    instruction: str | None = Field(default=None, description="SKILL.md指令正文")
    license: str | None = Field(default=None, description="SKILL.md frontmatter license")
    compatibility: str | None = Field(default=None, description="SKILL.md frontmatter compatibility")
    metadata: dict | None = Field(default=None, description="SKILL.md frontmatter metadata")
    allowedTools: str | None = Field(default=None, description="SKILL.md frontmatter allowed-tools")
    files: list[SkillFileVO] = Field(default_factory=list, description="SKILL目录内资源文件清单")
    status: int = Field(description="启停状态(0:禁用;1:启用)")
    source: str = Field(description="来源(builtin/admin)")
    agentCount: int = Field(default=0, description="被Agent关联数")
    marketShared: int = Field(default=0, description="是否共享至Skill市场(0:否;1:是)")
    createTime: datetime | None = Field(default=None, description="创建时间")
    updateTime: datetime | None = Field(default=None, description="更新时间")


class SkillListItem(OrmResult):
    """Skill 列表项（不含全文，渐进式加载不注入 instruction）"""

    id: int = Field(description="主键")
    name: str = Field(description="Skill名称")
    description: str = Field(description="Skill描述")
    scene: str = Field(default="", description="适用场景")
    status: int = Field(description="启停状态(0:禁用;1:启用)")
    source: str = Field(description="来源(builtin/admin)")
    marketShared: int = Field(default=0, description="是否共享至Skill市场(0:否;1:是)")
    agentCount: int = Field(default=0, description="被Agent关联数")
    createTime: datetime | None = Field(default=None, description="创建时间")
    updateTime: datetime | None = Field(default=None, description="更新时间")


class SkillTestForm(OrmResult):
    """试运行 Skill 请求体（测试数据不入库不推送）"""

    inputData: Any | None = Field(default=None, description="测试输入数据")


class SkillShareForm(OrmResult):
    """共享 Skill 至市场请求体"""

    skillId: int = Field(..., description="Skill主键")


class SkillMarketVO(OrmResult):
    """Skill 市场目录项"""

    skillId: int = Field(description="Skill主键")
    name: str = Field(description="Skill名称")
    description: str = Field(description="Skill描述")
    enabled: bool = Field(description="是否已启用")
    agentCount: int = Field(default=0, description="已关联Agent数")


class SkillPageQuery(BasePageQuery):
    """Skill 列表查询参数"""

    keyword: str | None = Field(default=None, description="关键字(按名称模糊搜索)")
