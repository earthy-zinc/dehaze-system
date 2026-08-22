"""Skills 管理 Schema 模型（F-M08-006）。

请求/响应字段采用 snake_case 定义、序列化输出 camelCase（继承 OrmResult），
与 API 契约（API接口.md Skills 管理接口）对齐。
"""

from datetime import datetime

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
    """创建 Skill 请求体"""

    name: str = Field(..., min_length=1, max_length=128, description="Skill名称(唯一)")
    description: str = Field(..., min_length=1, max_length=500, description="Skill描述")
    content: str = Field(..., min_length=1, description="Markdown指令全文")

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
    content: str | None = Field(default=None, min_length=1, description="Markdown指令全文")

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
    """启停 Skill 请求体"""

    enabled: bool = Field(..., description="目标启停状态(true:启用;false:禁用)")


class SkillResult(OrmResult):
    """Skill 详情（管理员全部字段）"""

    id: int = Field(description="主键")
    name: str = Field(description="Skill名称")
    description: str = Field(description="Skill描述")
    content: str | None = Field(default=None, description="Markdown指令全文")
    status: int = Field(description="启停状态(1:启用;2:禁用)")
    source: str = Field(description="来源(builtin/admin)")
    createTime: datetime | None = Field(default=None, description="创建时间")
    updateTime: datetime | None = Field(default=None, description="更新时间")


class SkillListItem(OrmResult):
    """Skill 列表项（不含全文，渐进式加载不注入 content）"""

    id: int = Field(description="主键")
    name: str = Field(description="Skill名称")
    description: str = Field(description="Skill描述")
    status: int = Field(description="启停状态(1:启用;2:禁用)")
    source: str = Field(description="来源(builtin/admin)")
    createTime: datetime | None = Field(default=None, description="创建时间")
    updateTime: datetime | None = Field(default=None, description="更新时间")


class SkillPageQuery(BasePageQuery):
    """Skill 列表查询参数"""

    keyword: str | None = Field(default=None, description="关键字(按名称模糊搜索)")
