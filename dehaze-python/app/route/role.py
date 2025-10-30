from flasgger import swag_from
from flask import Blueprint, request

from app.service.role import RoleService
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error

role_blueprint = Blueprint('role', __name__, url_prefix='/api/v1/roles')


@role_blueprint.route('/page', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['角色管理'],
    'summary': '获取角色分页列表',
    'description': '获取角色分页列表',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'pageNum',
            'in': 'query',
            'required': False,
            'schema': {'type': 'integer', 'default': 1}
        },
        {
            'name': 'pageSize',
            'in': 'query',
            'required': False,
            'schema': {'type': 'integer', 'default': 10}
        },
        {
            'name': 'keywords',
            'in': 'query',
            'required': False,
            'schema': {'type': 'string'}
        }
    ],
    'responses': {
        '200': {
            'description': '获取成功'
        }
    }
})
def get_role_page():
    """获取角色分页列表"""
    page = request.args.get('pageNum', 1, type=int)
    page_size = request.args.get('pageSize', 10, type=int)
    keywords = request.args.get('keywords', type=str)

    roles, total = RoleService.get_role_list(page, page_size, keywords)

    role_list = []
    for role in roles:
        role_list.append({
            'id': role.id,
            'name': role.name,
            'code': role.code,
            'sort': role.sort,
            'status': role.status,
            'dataScope': role.data_scope,
            'createTime': role.create_time.isoformat() if role.create_time else None
        })

    return success({
        'list': role_list,
        'total': total,
        'pageNum': page,
        'pageSize': page_size
    })


@role_blueprint.route('/options', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['角色管理'],
    'summary': '获取角色下拉列表',
    'description': '获取角色下拉列表',
    'security': [{'BearerAuth': []}],
    'responses': {
        '200': {
            'description': '获取成功'
        }
    }
})
def list_role_options():
    """角色下拉列表"""
    options = RoleService.get_role_options()
    return success(options)


@role_blueprint.route('/', methods=['POST'])
@jwt_required
@swag_from({
    'tags': ['角色管理'],
    'summary': '新增角色',
    'description': '新增角色',
    'security': [{'BearerAuth': []}],
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': '角色名称'
                        },
                        'code': {
                            'type': 'string',
                            'description': '角色编码'
                        },
                        'sort': {
                            'type': 'integer',
                            'description': '排序'
                        },
                        'status': {
                            'type': 'integer',
                            'description': '状态(1-正常；0-停用)'
                        },
                        'dataScope': {
                            'type': 'integer',
                            'description': '数据权限'
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '新增成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def add_role():
    """新增角色"""
    data = request.get_json()
    result = RoleService.create_role(data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '新增成功')


@role_blueprint.route('/<int:role_id>/form', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['角色管理'],
    'summary': '获取角色表单数据',
    'description': '获取角色表单数据',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'role_id',
            'in': 'path',
            'required': True,
            'schema': {'type': 'integer'}
        }
    ],
    'responses': {
        '200': {
            'description': '获取成功'
        },
        '404': {
            'description': '角色不存在'
        }
    }
})
def get_role_form(role_id):
    """获取角色表单数据"""
    role = RoleService.get_role_by_id(role_id)

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


@role_blueprint.route('/<int:role_id>', methods=['PUT'])
@jwt_required
@swag_from({
    'tags': ['角色管理'],
    'summary': '修改角色',
    'description': '修改角色',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'role_id',
            'in': 'path',
            'required': True,
            'schema': {'type': 'integer'}
        }
    ],
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': '角色名称'
                        },
                        'code': {
                            'type': 'string',
                            'description': '角色编码'
                        },
                        'sort': {
                            'type': 'integer',
                            'description': '排序'
                        },
                        'status': {
                            'type': 'integer',
                            'description': '状态(1-正常；0-停用)'
                        },
                        'dataScope': {
                            'type': 'integer',
                            'description': '数据权限'
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '更新成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def update_role(role_id):
    """修改角色"""
    data = request.get_json()
    result = RoleService.update_role(role_id, data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '更新成功')


@role_blueprint.route('/<string:ids>', methods=['DELETE'])
@jwt_required
@swag_from({
    'tags': ['角色管理'],
    'summary': '删除角色',
    'description': '删除角色',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'ids',
            'in': 'path',
            'required': True,
            'schema': {'type': 'string'},
            'description': '角色ID，多个以英文逗号分隔'
        }
    ],
    'responses': {
        '200': {
            'description': '删除成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def delete_roles(ids):
    """删除角色"""
    result = RoleService.delete_roles(ids)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '删除成功')


@role_blueprint.route('/<int:role_id>/status', methods=['PUT'])
@jwt_required
@swag_from({
    'tags': ['角色管理'],
    'summary': '修改角色状态',
    'description': '修改角色状态',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'role_id',
            'in': 'path',
            'required': True,
            'schema': {'type': 'integer'}
        },
        {
            'name': 'status',
            'in': 'query',
            'required': True,
            'schema': {'type': 'integer', 'enum': [0, 1]},
            'description': '状态(1-启用；0-停用)'
        }
    ],
    'responses': {
        '200': {
            'description': '更新成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def update_role_status(role_id):
    """修改角色状态"""
    status = request.args.get('status', type=int)

    result = RoleService.update_role_status(role_id, status)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '更新成功')


@role_blueprint.route('/<int:role_id>/menuIds', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['角色管理'],
    'summary': '获取角色的菜单ID集合',
    'description': '获取角色的菜单ID集合',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'role_id',
            'in': 'path',
            'required': True,
            'schema': {'type': 'integer'}
        }
    ],
    'responses': {
        '200': {
            'description': '获取成功'
        },
        '404': {
            'description': '角色不存在'
        }
    }
})
def get_role_menu_ids(role_id):
    """获取角色的菜单ID集合"""
    role = RoleService.get_role_by_id(role_id)

    if not role:
        return error('角色不存在', 404)

    menu_ids = RoleService.get_role_menu_ids(role_id)
    return success(menu_ids)


@role_blueprint.route('/<int:role_id>/menus', methods=['PUT'])
@jwt_required
@swag_from({
    'tags': ['角色管理'],
    'summary': '分配菜单给角色',
    'description': '分配菜单给角色',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'role_id',
            'in': 'path',
            'required': True,
            'schema': {'type': 'integer'}
        }
    ],
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'array',
                    'items': {'type': 'integer'}
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '分配成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def assign_menus_to_role(role_id):
    """分配菜单给角色"""
    menu_ids = request.get_json()

    result = RoleService.assign_menus_to_role(role_id, menu_ids)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '分配成功')
