"""
角色管理路由 - 使用 flask-openapi3 自动生成 Swagger 文档
"""
from flask_openapi3 import APIBlueprint, Tag

from app.models.schema.role import (
    RolePageQuery,
    StatusQuery,
    RoleIdPath,
    RoleIdsPath,
    RoleForm,
    MenuIdsBody,
    RolePageVO,
)
from app.service.role import RoleService
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error


# 定义 Tag
role_tag = Tag(name="角色管理", description="角色相关接口")

# 创建 APIBlueprint（自动携带安全配置）
role_blueprint = APIBlueprint(
    "role",
    __name__,
    url_prefix="/api/v1/roles",
    abp_tags=[role_tag],
    abp_security=[{"BearerAuth": []}]
)


# ==================== 数据权限映射 ====================
DATA_SCOPE_LABELS = {
    0: '全部数据',
    1: '部门及子部门数据',
    2: '本部门数据',
    3: '本人数据'
}


# ==================== 路由定义 ====================

@role_blueprint.get(
    "/page",
    summary="获取角色分页列表",
    description="根据关键词查询角色分页列表"
)
@jwt_required
def get_role_page(query: RolePageQuery):
    """获取角色分页列表"""
    roles, total = RoleService.get_role_list(query.pageNum, query.pageSize, query.keywords)

    role_list = []
    for role in roles:
        role_list.append({
            'id': role.id,
            'name': role.name,
            'code': role.code,
            'sort': role.sort,
            'status': role.status,
            'dataScope': role.data_scope,
            'dataScopeLabel': DATA_SCOPE_LABELS.get(role.data_scope, ''),
            'createTime': role.create_time.isoformat() if role.create_time else None
        })

    return success({
        'list': role_list,
        'total': total,
        'pageNum': query.pageNum,
        'pageSize': query.pageSize
    })


@role_blueprint.get(
    "/options",
    summary="获取角色下拉列表",
    description="获取所有角色的下拉选项列表"
)
@jwt_required
def list_role_options():
    """角色下拉列表"""
    options = RoleService.get_role_options()
    return success(options)


@role_blueprint.post(
    "/",
    summary="新增角色",
    description="创建一个新的角色"
)
@jwt_required
def add_role(body: RoleForm):
    """新增角色"""
    data = body.model_dump(exclude_none=True)
    # 转换字段名 dataScope -> data_scope
    if 'dataScope' in data:
        data['data_scope'] = data.pop('dataScope')
    
    result = RoleService.create_role(data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '新增成功')


@role_blueprint.get(
    "/<int:role_id>/form",
    summary="获取角色表单数据",
    description="根据角色ID获取角色的表单数据"
)
@jwt_required
def get_role_form(path: RoleIdPath):
    """获取角色表单数据"""
    role = RoleService.get_role_by_id(path.role_id)

    if not role:
        return error('角色不存在', 404)

    return success({
        'id': role.id,
        'name': role.name,
        'code': role.code,
        'sort': role.sort,
        'status': role.status,
        'dataScope': role.data_scope
    })


@role_blueprint.put(
    "/<int:role_id>",
    summary="修改角色",
    description="根据角色ID修改角色信息"
)
@jwt_required
def update_role(path: RoleIdPath, body: RoleForm):
    """修改角色"""
    data = body.model_dump(exclude_none=True)
    # 转换字段名 dataScope -> data_scope
    if 'dataScope' in data:
        data['data_scope'] = data.pop('dataScope')
    
    result = RoleService.update_role(path.role_id, data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '更新成功')


@role_blueprint.delete(
    "/<ids>",
    summary="删除角色",
    description="批量删除角色，多个ID以英文逗号分隔"
)
@jwt_required
def delete_roles(path: RoleIdsPath):
    """删除角色"""
    result = RoleService.delete_roles(path.ids)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '删除成功')


@role_blueprint.put(
    "/<int:role_id>/status",
    summary="修改角色状态",
    description="启用或停用角色"
)
@jwt_required
def update_role_status(path: RoleIdPath, query: StatusQuery):
    """修改角色状态"""
    result = RoleService.update_role_status(path.role_id, query.status)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '更新成功')


@role_blueprint.get(
    "/<int:role_id>/menuIds",
    summary="获取角色的菜单ID集合",
    description="获取角色拥有的所有菜单ID"
)
@jwt_required
def get_role_menu_ids(path: RoleIdPath):
    """获取角色的菜单ID集合"""
    role = RoleService.get_role_by_id(path.role_id)

    if not role:
        return error('角色不存在', 404)

    menu_ids = RoleService.get_role_menu_ids(path.role_id)
    return success(menu_ids)


@role_blueprint.put(
    "/<int:role_id>/menus",
    summary="分配菜单给角色",
    description="为角色分配菜单权限（包括按钮权限）"
)
@jwt_required
def assign_menus_to_role(path: RoleIdPath, body: MenuIdsBody):
    """分配菜单给角色"""
    # RootModel 使用 .root 访问实际的列表数据
    menu_ids = body.root if hasattr(body, 'root') else body
    result = RoleService.assign_menus_to_role(path.role_id, menu_ids)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '分配成功')
