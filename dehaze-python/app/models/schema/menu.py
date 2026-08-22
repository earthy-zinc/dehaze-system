"""
菜单模块 Schema 模型
"""

from pydantic import BaseModel, Field, field_validator

from app.models.schema.common import validate_no_xss

# 菜单类型字符串枚举 → 整数映射（对齐 Java MenuTypeEnum）
MENU_TYPE_NAME_TO_VALUE = {
    "MENU": 1,
    "CATALOG": 2,
    "EXTLINK": 3,
    "BUTTON": 4,
}


# ==================== 查询参数模型 ====================


class MenuVisibleBody(BaseModel):
    """菜单显示状态请求体"""

    visible: int = Field(..., ge=0, le=1, description="显示状态(1:显示;0:隐藏)")


# ==================== 路径参数模型 ====================


# ==================== 请求体模型 ====================


class MenuForm(BaseModel):
    """菜单表单"""

    id: int | None = Field(default=None, description="菜单ID")
    parentId: int | None = Field(default=None, description="父菜单ID")
    name: str = Field(..., min_length=1, max_length=64, description="菜单名称")
    type: int = Field(..., ge=1, le=4, description="菜单类型(1-菜单；2-目录；3-外链；4-按钮权限)")
    path: str | None = Field(default=None, max_length=200, description="路由路径")
    component: str | None = Field(
        default=None, max_length=200, description="组件路径(vue页面完整路径，省略.vue后缀)"
    )
    perm: str | None = Field(default=None, max_length=100, description="权限标识")
    visible: int | None = Field(default=1, ge=0, le=1, description="显示状态(1:显示;0:隐藏)")
    sort: int | None = Field(default=0, ge=0, description="排序(数字越小排名越靠前)")
    icon: str | None = Field(default=None, max_length=100, description="菜单图标")
    redirect: str | None = Field(default=None, max_length=200, description="跳转路径")
    keepAlive: int | None = Field(default=None, ge=0, le=1, description="【菜单】是否开启页面缓存")
    alwaysShow: int | None = Field(
        default=None, ge=0, le=1, description="【目录】只有一个子路由是否始终显示"
    )

    name_no_xss_validator = field_validator("name")(validate_no_xss)

    @field_validator("type", mode="before")
    @classmethod
    def convert_type_from_enum_name(cls, v):
        """支持字符串枚举名（CATALOG/MENU/EXTLINK/BUTTON），对齐 Java MenuTypeEnum 序列化"""
        if isinstance(v, str) and v in MENU_TYPE_NAME_TO_VALUE:
            return MENU_TYPE_NAME_TO_VALUE[v]
        return v


# ==================== 响应模型 ====================


class MenuVO(BaseModel):
    """菜单视图对象"""

    id: int = Field(description="菜单ID")
    parentId: int = Field(description="父菜单ID")
    name: str = Field(description="菜单名称")
    type: str = Field(description="菜单类型(MENU/CATALOG/EXTLINK/BUTTON)")
    path: str | None = Field(default=None, description="路由路径")
    component: str | None = Field(default=None, description="组件路径")
    sort: int = Field(description="菜单排序(数字越小排名越靠前)")
    visible: int = Field(description="菜单是否可见(1:显示;0:隐藏)")
    icon: str | None = Field(default=None, description="ICON")
    redirect: str | None = Field(default=None, description="跳转路径")
    perm: str | None = Field(default=None, description="按钮权限标识")
    children: list["MenuVO"] | None = Field(default=None, description="子菜单")


class RouteMeta(BaseModel):
    """路由元数据（对齐 Java RouteVO.Meta）"""

    title: str = Field(description="路由标题")
    icon: str | None = Field(default=None, description="图标")
    hidden: bool = Field(default=False, description="是否隐藏")
    keepAlive: bool | None = Field(default=None, description="是否开启页面缓存")
    alwaysShow: bool | None = Field(default=None, description="目录是否始终显示")


class RouteVO(BaseModel):
    """路由视图对象（对齐 Java/Go RouteVO）"""

    name: str = Field(description="路由名称")
    path: str = Field(description="路由路径")
    component: str | None = Field(default=None, description="组件路径")
    redirect: str | None = Field(default=None, description="跳转路径")
    meta: RouteMeta = Field(description="路由元数据")
    children: list["RouteVO"] | None = Field(default=None, description="子路由")


class MenuOptionVO(BaseModel):
    """菜单选项视图对象（对齐 Java Option<T>）"""

    value: int = Field(description="选项值（菜单ID）")
    label: str = Field(description="选项标签（菜单名称）")
    children: list["MenuOptionVO"] | None = Field(default=None, description="子选项列表")


class MenuFormVO(BaseModel):
    """菜单表单VO - 用于编辑回显"""

    id: int = Field(description="菜单ID")
    parentId: int = Field(default=0, description="父菜单ID")
    name: str = Field(description="菜单名称")
    type: str = Field(description="菜单类型(MENU/CATALOG/EXTLINK/BUTTON)")
    path: str | None = Field(default=None, description="路由路径")
    component: str | None = Field(default=None, description="组件路径")
    perm: str | None = Field(default=None, description="权限标识")
    visible: int = Field(default=1, description="显示状态(1:显示;0:隐藏)")
    sort: int = Field(default=0, description="排序")
    icon: str | None = Field(default=None, description="菜单图标")
    redirect: str | None = Field(default=None, description="跳转路径")
    keepAlive: int | None = Field(default=None, description="是否开启页面缓存")
    alwaysShow: int | None = Field(default=None, description="只有一个子路由是否始终显示")


# 重建模型以处理自引用
MenuVO.model_rebuild()
RouteVO.model_rebuild()
