"""
菜单模块 Schema 模型
"""
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


# ==================== 查询参数模型 ====================

class MenuQuery(BaseModel):
    """菜单查询参数"""
    keywords: Optional[str] = Field(default=None, description="关键词(菜单名称)")
    status: Optional[int] = Field(
        default=None, ge=0, le=1, description="状态(1-显示；0-隐藏)")


class MenuVisibleQuery(BaseModel):
    """菜单显示状态查询参数"""
    visible: int = Field(..., ge=0, le=1, description="显示状态(1:显示;0:隐藏)")


class MenuVisibleBody(BaseModel):
    """菜单显示状态请求体"""
    visible: int = Field(..., ge=0, le=1, description="显示状态(1:显示;0:隐藏)")


# ==================== 路径参数模型 ====================

class MenuIdPath(BaseModel):
    """菜单ID路径参数"""
    menu_id: int = Field(..., description="菜单ID")


# ==================== 请求体模型 ====================

class MenuForm(BaseModel):
    """菜单表单"""
    id: Optional[int] = Field(default=None, description="菜单ID")
    parentId: Optional[int] = Field(default=None, description="父菜单ID")
    name: str = Field(..., min_length=1, max_length=100, description="菜单名称")
    type: int = Field(..., ge=1, le=4,
                      description="菜单类型(1-菜单；2-目录；3-外链；4-按钮权限)")
    path: Optional[str] = Field(
        default=None, max_length=200, description="路由路径")
    component: Optional[str] = Field(
        default=None, max_length=200, description="组件路径(vue页面完整路径，省略.vue后缀)")
    perm: Optional[str] = Field(
        default=None, max_length=100, description="权限标识")
    visible: Optional[int] = Field(
        default=1, ge=0, le=1, description="显示状态(1:显示;0:隐藏)")
    sort: Optional[int] = Field(default=0, ge=0, description="排序(数字越小排名越靠前)")
    icon: Optional[str] = Field(
        default=None, max_length=100, description="菜单图标")
    redirect: Optional[str] = Field(
        default=None, max_length=200, description="跳转路径")
    keepAlive: Optional[int] = Field(
        default=None, ge=0, le=1, description="【菜单】是否开启页面缓存")
    alwaysShow: Optional[int] = Field(
        default=None, ge=0, le=1, description="【目录】只有一个子路由是否始终显示")


# ==================== 响应模型 ====================

class MenuVO(BaseModel):
    """菜单视图对象"""
    id: int = Field(description="菜单ID")
    parentId: int = Field(description="父菜单ID")
    name: str = Field(description="菜单名称")
    type: int = Field(description="菜单类型(1-菜单；2-目录；3-外链；4-按钮权限)")
    path: Optional[str] = Field(default=None, description="路由路径")
    component: Optional[str] = Field(default=None, description="组件路径")
    sort: int = Field(description="菜单排序(数字越小排名越靠前)")
    visible: int = Field(description="菜单是否可见(1:显示;0:隐藏)")
    icon: Optional[str] = Field(default=None, description="ICON")
    redirect: Optional[str] = Field(default=None, description="跳转路径")
    perm: Optional[str] = Field(default=None, description="按钮权限标识")
    children: Optional[List["MenuVO"]] = Field(default=None, description="子菜单")


class RouteVO(BaseModel):
    """路由视图对象"""
    id: int = Field(description="菜单ID")
    parentId: int = Field(description="父菜单ID")
    name: str = Field(description="菜单名称")
    type: int = Field(description="菜单类型(1-菜单；2-目录；3-外链；4-按钮权限)")
    path: Optional[str] = Field(default=None, description="路由路径")
    component: Optional[str] = Field(default=None, description="组件路径")
    sort: int = Field(description="菜单排序(数字越小排名越靠前)")
    visible: int = Field(description="菜单是否可见(1:显示;0:隐藏)")
    icon: Optional[str] = Field(default=None, description="ICON")
    redirect: Optional[str] = Field(default=None, description="跳转路径")
    perm: Optional[str] = Field(default=None, description="按钮权限标识")
    children: Optional[List["RouteVO"]] = Field(
        default=None, description="子菜单")


class MenuOptionVO(BaseModel):
    """菜单选项视图对象"""
    id: int = Field(description="菜单ID")
    name: str = Field(description="菜单名称")


class MenuFormVO(BaseModel):
    """菜单表单VO - 用于编辑回显"""
    id: int = Field(description="菜单ID")
    parentId: int = Field(default=0, description="父菜单ID")
    name: str = Field(description="菜单名称")
    type: int = Field(description="菜单类型(1-菜单；2-目录；3-外链；4-按钮权限)")
    path: Optional[str] = Field(default=None, description="路由路径")
    component: Optional[str] = Field(default=None, description="组件路径")
    perm: Optional[str] = Field(default=None, description="权限标识")
    visible: int = Field(default=1, description="显示状态(1:显示;0:隐藏)")
    sort: int = Field(default=0, description="排序")
    icon: Optional[str] = Field(default=None, description="菜单图标")
    redirect: Optional[str] = Field(default=None, description="跳转路径")
    keepAlive: Optional[int] = Field(default=None, description="是否开启页面缓存")
    alwaysShow: Optional[int] = Field(
        default=None, description="只有一个子路由是否始终显示")


# 重建模型以处理自引用
MenuVO.model_rebuild()
RouteVO.model_rebuild()
