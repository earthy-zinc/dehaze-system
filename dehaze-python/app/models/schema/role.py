"""
角色模块 Schema 模型
"""

from pydantic import BaseModel, Field, RootModel, field_validator

from app.models.schema.common import BasePageQuery, validate_no_xss

# ==================== 查询参数模型 ====================


class RolePageQuery(BasePageQuery):
    """角色分页查询参数"""

    keywords: str | None = Field(default=None, description="关键词(角色名称/角色编码)")


# ==================== 路径参数模型 ====================


# ==================== 请求体模型 ====================


class RoleForm(BaseModel):
    """角色表单"""

    id: int | None = Field(default=None, description="角色ID")
    name: str = Field(..., min_length=1, max_length=64, description="角色名称")
    code: str = Field(..., min_length=1, max_length=32, description="角色编码")
    sort: int = Field(default=0, ge=0, description="排序")
    status: int = Field(default=1, ge=0, le=1, description="状态(1-正常；0-停用)")
    dataScope: int | None = Field(
        default=None,
        ge=0,
        le=3,
        description="数据权限(0-全部数据；1-部门及子部门数据；2-本部门数据；3-本人数据)；创建时必填",
    )

    name_code_no_xss_validator = field_validator("name", "code")(validate_no_xss)


class MenuIdsBody(RootModel[list[int]]):
    """菜单ID列表请求体 - 使用 RootModel 包装列表"""

    root: list[int] = Field(..., description="菜单ID列表")


# ==================== 响应模型 ====================


class RolePageVO(BaseModel):
    """角色分页VO"""

    id: int = Field(description="角色ID")
    name: str = Field(description="角色名称")
    code: str = Field(description="角色编码")
    sort: int = Field(description="排序")
    status: int = Field(description="状态(1-正常；0-停用)")
    dataScope: int = Field(description="数据权限")
    dataScopeLabel: str | None = Field(default=None, description="数据权限标签")
    createTime: str | None = Field(default=None, description="创建时间")


class RoleOptionVO(BaseModel):
    """角色下拉选项VO"""

    value: int = Field(description="选项值(角色ID)")
    label: str = Field(description="选项标签(角色名称)")


class RoleFormVO(BaseModel):
    """角色表单VO"""

    id: int = Field(description="角色ID")
    name: str = Field(description="角色名称")
    code: str = Field(description="角色编码")
    sort: int = Field(description="排序")
    status: int = Field(description="状态(1-正常；0-停用)")
    dataScope: int = Field(description="数据权限")
    dataScopeLabel: str | None = Field(default=None, description="数据权限标签")
    createTime: str | None = Field(default=None, description="创建时间")
