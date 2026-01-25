"""
菜单管理路由 - 使用 flask-openapi3 自动生成 Swagger 文档
"""
from flask_openapi3 import APIBlueprint, Tag

from app.models.schema.menu import (
    MenuQuery,
    MenuIdPath,
    MenuVisibleQuery,
    MenuForm,
)
from app.service.menu import MenuService
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error


# 定义 Tag
menu_tag = Tag(name="菜单管理", description="菜单相关接口")

# 创建 APIBlueprint（自动携带安全配置）
menu_blueprint = APIBlueprint(
    "menu",
    __name__,
    url_prefix="/api/v1/menus",
    abp_tags=[menu_tag],
    abp_security=[{"BearerAuth": []}]
)


# ==================== 路由定义 ====================

@menu_blueprint.get(
    "/",
    summary="菜单列表",
    description="获取菜单列表（树形结构）"
)
@jwt_required
def list_menus(query: MenuQuery):
    """获取菜单列表"""
    menu_list = MenuService.list_menus(query.keywords)
    return success(menu_list)


@menu_blueprint.get(
    "/options",
    summary="菜单下拉列表",
    description="获取菜单下拉列表"
)
@jwt_required
def list_menu_options():
    """菜单下拉列表"""
    options = MenuService.list_menu_options()
    return success(options)


@menu_blueprint.get(
    "/routes",
    summary="路由列表",
    description="获取路由列表"
)
@jwt_required
def list_routes():
    """路由列表"""
    route_list = MenuService.list_routes()
    return success(route_list)


@menu_blueprint.get(
    "/<int:menu_id>/form",
    summary="菜单表单数据",
    description="获取菜单表单数据"
)
@jwt_required
def get_menu_form(path: MenuIdPath):
    """获取菜单表单数据"""
    menu_form = MenuService.get_menu_form(path.menu_id)

    if menu_form is None:
        return error('菜单不存在', 404)

    return success(menu_form)


@menu_blueprint.post(
    "/",
    summary="新增菜单",
    description="新增菜单"
)
@jwt_required
def add_menu(body: MenuForm):
    """新增菜单"""
    data = body.model_dump(exclude_none=True)
    result = MenuService.save_menu(data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '保存成功')


@menu_blueprint.put(
    "/<int:menu_id>",
    summary="修改菜单",
    description="修改菜单"
)
@jwt_required
def update_menu(path: MenuIdPath, body: MenuForm):
    """修改菜单"""
    data = body.model_dump(exclude_none=True)
    data['id'] = path.menu_id
    result = MenuService.save_menu(data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '保存成功')


@menu_blueprint.delete(
    "/<int:menu_id>",
    summary="删除菜单",
    description="删除菜单"
)
@jwt_required
def delete_menu(path: MenuIdPath):
    """删除菜单"""
    result = MenuService.delete_menu(path.menu_id)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '删除成功')


@menu_blueprint.patch(
    "/<int:menu_id>",
    summary="修改菜单显示状态",
    description="修改菜单显示状态"
)
@jwt_required
def update_menu_visible(path: MenuIdPath, query: MenuVisibleQuery):
    """修改菜单显示状态"""
    result = MenuService.update_menu_visible(path.menu_id, query.visible)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '更新成功')
