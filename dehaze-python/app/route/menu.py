from flasgger import swag_from
from flask import Blueprint, request

from app.service.menu import MenuService
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error

menu_blueprint = Blueprint('menu', __name__, url_prefix='/api/v1/menus')


@menu_blueprint.route('/', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['菜单管理'],
    'summary': '菜单列表',
    'description': '获取菜单列表（树形结构）',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'keywords',
            'in': 'query',
            'required': False,
            'schema': {'type': 'string'},
            'description': '关键字（菜单名称）'
        }
    ],
    'responses': {
        '200': {
            'description': '获取成功'
        }
    }
})
def list_menus():
    """获取菜单列表"""
    keywords = request.args.get('keywords', type=str)

    menu_list = MenuService.list_menus(keywords)
    return success(menu_list)


@menu_blueprint.route('/options', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['菜单管理'],
    'summary': '菜单下拉列表',
    'description': '获取菜单下拉列表',
    'security': [{'BearerAuth': []}],
    'responses': {
        '200': {
            'description': '获取成功'
        }
    }
})
def list_menu_options():
    """菜单下拉列表"""
    options = MenuService.list_menu_options()
    return success(options)


@menu_blueprint.route('/routes', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['菜单管理'],
    'summary': '路由列表',
    'description': '获取路由列表',
    'security': [{'BearerAuth': []}],
    'responses': {
        '200': {
            'description': '获取成功'
        }
    }
})
def list_routes():
    """路由列表"""
    route_list = MenuService.list_routes()
    return success(route_list)


@menu_blueprint.route('/<int:menu_id>/form', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['菜单管理'],
    'summary': '菜单表单数据',
    'description': '获取菜单表单数据',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'menu_id',
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
            'description': '菜单不存在'
        }
    }
})
def get_menu_form(menu_id):
    """获取菜单表单数据"""
    menu_form = MenuService.get_menu_form(menu_id)

    if menu_form is None:
        return error('菜单不存在', 404)

    return success(menu_form)


@menu_blueprint.route('/', methods=['POST'])
@jwt_required
@swag_from({
    'tags': ['菜单管理'],
    'summary': '新增菜单',
    'description': '新增菜单',
    'security': [{'BearerAuth': []}],
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'parentId': {'type': 'integer', 'description': '父菜单ID'},
                        'name': {'type': 'string', 'description': '菜单名称'},
                        'type': {'type': 'integer', 'description': '菜单类型(1:菜单 2:目录 3:外链 4:按钮)'},
                        'path': {'type': 'string', 'description': '路由路径'},
                        'component': {'type': 'string', 'description': '组件路径'},
                        'perm': {'type': 'string', 'description': '权限标识'},
                        'visible': {'type': 'integer', 'description': '显示状态(1:显示;0:隐藏)'},
                        'sort': {'type': 'integer', 'description': '排序'},
                        'icon': {'type': 'string', 'description': '菜单图标'},
                        'redirect': {'type': 'string', 'description': '跳转路径'},
                        'alwaysShow': {'type': 'integer', 'description': '目录只有一个子路由是否始终显示'},
                        'keepAlive': {'type': 'integer', 'description': '菜单是否开启页面缓存'}
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '保存成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def add_menu():
    """新增菜单"""
    data = request.get_json()
    result = MenuService.save_menu(data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '保存成功')


@menu_blueprint.route('/<int:menu_id>', methods=['PUT'])
@jwt_required
@swag_from({
    'tags': ['菜单管理'],
    'summary': '修改菜单',
    'description': '修改菜单',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'menu_id',
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
                        'parentId': {'type': 'integer', 'description': '父菜单ID'},
                        'name': {'type': 'string', 'description': '菜单名称'},
                        'type': {'type': 'integer', 'description': '菜单类型(1:菜单 2:目录 3:外链 4:按钮)'},
                        'path': {'type': 'string', 'description': '路由路径'},
                        'component': {'type': 'string', 'description': '组件路径'},
                        'perm': {'type': 'string', 'description': '权限标识'},
                        'visible': {'type': 'integer', 'description': '显示状态(1:显示;0:隐藏)'},
                        'sort': {'type': 'integer', 'description': '排序'},
                        'icon': {'type': 'string', 'description': '菜单图标'},
                        'redirect': {'type': 'string', 'description': '跳转路径'},
                        'alwaysShow': {'type': 'integer', 'description': '目录只有一个子路由是否始终显示'},
                        'keepAlive': {'type': 'integer', 'description': '菜单是否开启页面缓存'}
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '保存成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def update_menu(menu_id):
    """修改菜单"""
    data = request.get_json()
    data['id'] = menu_id
    result = MenuService.save_menu(data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '保存成功')


@menu_blueprint.route('/<int:menu_id>', methods=['DELETE'])
@jwt_required
@swag_from({
    'tags': ['菜单管理'],
    'summary': '删除菜单',
    'description': '删除菜单',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'menu_id',
            'in': 'path',
            'required': True,
            'schema': {'type': 'integer'}
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
def delete_menu(menu_id):
    """删除菜单"""
    result = MenuService.delete_menu(menu_id)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '删除成功')


@menu_blueprint.route('/<int:menu_id>', methods=['PATCH'])
@jwt_required
@swag_from({
    'tags': ['菜单管理'],
    'summary': '修改菜单显示状态',
    'description': '修改菜单显示状态',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'menu_id',
            'in': 'path',
            'required': True,
            'schema': {'type': 'integer'}
        },
        {
            'name': 'visible',
            'in': 'query',
            'required': True,
            'schema': {'type': 'integer', 'enum': [0, 1]},
            'description': '显示状态(1:显示;0:隐藏)'
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
def update_menu_visible(menu_id):
    """修改菜单显示状态"""
    visible = request.args.get('visible', type=int)

    result = MenuService.update_menu_visible(menu_id, visible)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '更新成功')
