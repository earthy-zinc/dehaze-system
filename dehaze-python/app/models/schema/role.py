"""
角色模块 Schema 模型
"""
from typing import List, Optional

from app.models.schema.common import BasePageQuery
from pydantic import BaseModel, Field, RootModel

# ==================== 查询参数模型 ====================


class RolePageQuery(BasePageQuery):
    """角色分页查询参数"""
    keywords: Optional[str] = Field(default=None, description="关键词(角色名称/角色编码)")


class StatusQuery(BaseModel):
    """状态修改查询参数"""
    status: int = Field(..., ge=0, le=1, description="状态(1-启用；0-停用)")


# ==================== 路径参数模型 ====================

class RoleIdPath(BaseModel):
    """角色ID路径参数"""
    role_id: int = Field(..., description="角色ID")


class RoleIdsPath(BaseModel):
    """批量删除路径参数"""
    ids: str = Field(..., description="角色ID，多个以英文逗号(,)分隔")


# ==================== 请求体模型 ====================

class RoleForm(BaseModel):
    """角色表单"""
    id: Optional[int] = Field(default=None, description="角色ID")
    name: str = Field(..., min_length=1, max_length=50, description="角色名称")
    code: str = Field(..., min_length=1, max_length=50, description="角色编码")
    sort: int = Field(default=0, ge=0, description="排序")
    status: int = Field(default=1, ge=0, le=1, description="状态(1-正常；0-停用)")
    dataScope: int = Field(default=0, ge=0, le=3,
                           description="数据权限(0-全部数据；1-部门及子部门数据；2-本部门数据；3-本人数据)")


class MenuIdsBody(RootModel[List[int]]):
    """菜单ID列表请求体 - 使用 RootModel 包装列表"""
    root: List[int] = Field(..., description="菜单ID列表")


# ==================== 响应模型 ====================

class RolePageVO(BaseModel):
    """角色分页VO"""
    id: int = Field(description="角色ID")
    name: str = Field(description="角色名称")
    code: str = Field(description="角色编码")
    sort: int = Field(description="排序")
    status: int = Field(description="状态(1-正常；0-停用)")
    dataScope: int = Field(description="数据权限")
    dataScopeLabel: Optional[str] = Field(default=None, description="数据权限标签")
    createTime: Optional[str] = Field(default=None, description="创建时间")


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
    dataScopeLabel: Optional[str] = Field(default=None, description="数据权限标签")
    createTime: Optional[str] = Field(default=None, description="创建时间")
