from flask import Blueprint, request, current_app
from app.service.user import UserService
from app.utils.result import success, error
from app.utils.jwt_util import jwt_required, get_current_user_id
from flasgger import swag_from
import json

user_blueprint = Blueprint('user', __name__, url_prefix='/api/v1/users')


@user_blueprint.route('/login', methods=['POST'])
@swag_from({
    'tags': ['用户管理'],
    'summary': '用户登录',
    'description': '用户登录接口',
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'username': {
                            'type': 'string',
                            'description': '用户名'
                        },
                        'password': {
                            'type': 'string',
                            'description': '密码'
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '登录成功',
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'code': {'type': 'string'},
                            'msg': {'type': 'string'},
                            'data': {
                                'type': 'object',
                                'properties': {
                                    'token': {'type': 'string'},
                                    'user': {
                                        'type': 'object',
                                        'properties': {
                                            'id': {'type': 'integer'},
                                            'username': {'type': 'string'},
                                            'nickname': {'type': 'string'}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        '401': {
            'description': '认证失败'
        }
    }
})
def login():
    """用户登录"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return error('用户名和密码不能为空', 400)
    
    user = UserService.authenticate_user(username, password)
    if not user:
        return error('用户名或密码错误', 401)
    
    if user.status != 1:
        return error('用户已被禁用', 401)
    
    token = UserService.generate_token(user.id)
    
    return success({
        'token': token,
        'user': {
            'id': user.id,
            'username': user.username,
            'nickname': user.nickname
        }
    })


@user_blueprint.route('/register', methods=['POST'])
@swag_from({
    'tags': ['用户管理'],
    'summary': '用户注册',
    'description': '用户注册接口',
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'username': {
                            'type': 'string',
                            'description': '用户名'
                        },
                        'password': {
                            'type': 'string',
                            'description': '密码'
                        },
                        'nickname': {
                            'type': 'string',
                            'description': '昵称'
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '注册成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def register():
    """用户注册"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    nickname = data.get('nickname', username)
    
    if not username or not password:
        return error('用户名和密码不能为空', 400)
    
    # 检查用户名是否已存在
    existing_user = UserService.get_user_by_username(username)
    if existing_user:
        return error('用户名已存在', 400)
    
    # 创建用户
    user = UserService.create_user(username, password, nickname)
    
    return success({
        'id': user.id,
        'username': user.username,
        'nickname': user.nickname
    }, '注册成功')


@user_blueprint.route('/me', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['用户管理'],
    'summary': '获取当前用户信息',
    'description': '获取当前登录用户的信息',
    'security': [{'BearerAuth': []}],
    'responses': {
        '200': {
            'description': '获取成功',
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'code': {'type': 'string'},
                            'msg': {'type': 'string'},
                            'data': {
                                'type': 'object',
                                'properties': {
                                    'user': {
                                        'type': 'object',
                                        'properties': {
                                            'id': {'type': 'integer'},
                                            'username': {'type': 'string'},
                                            'nickname': {'type': 'string'},
                                            'roles': {
                                                'type': 'array',
                                                'items': {'type': 'string'}
                                            },
                                            'permissions': {
                                                'type': 'array',
                                                'items': {'type': 'string'}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
})
def get_current_user():
    """获取当前用户信息"""
    user_id = get_current_user_id()
    user = UserService.get_user_by_id(user_id)
    
    if not user:
        return error('用户不存在', 404)
    
    roles = UserService.get_user_roles(user_id)
    permissions = UserService.get_user_permissions(user_id)
    
    role_codes = [role.code for role in roles]
    
    return success({
        'user': {
            'id': user.id,
            'username': user.username,
            'nickname': user.nickname,
            'roles': role_codes,
            'permissions': permissions
        }
    })


@user_blueprint.route('/page', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['用户管理'],
    'summary': '获取用户分页列表',
    'description': '获取用户分页列表',
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
            'name': 'username',
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
def get_user_page():
    """获取用户分页列表"""
    page = request.args.get('pageNum', 1, type=int)
    page_size = request.args.get('pageSize', 10, type=int)
    username = request.args.get('username', type=str)
    
    users, total = UserService.get_user_list(page, page_size, username)
    
    user_list = []
    for user in users:
        user_list.append({
            'id': user.id,
            'username': user.username,
            'nickname': user.nickname,
            'gender': user.gender,
            'mobile': user.mobile,
            'email': user.email,
            'status': user.status,
            'createTime': user.create_time.isoformat() if user.create_time else None
        })
    
    return success({
        'list': user_list,
        'total': total,
        'pageNum': page,
        'pageSize': page_size
    })


@user_blueprint.route('/', methods=['POST'])
@jwt_required
@swag_from({
    'tags': ['用户管理'],
    'summary': '新增用户',
    'description': '新增用户接口',
    'security': [{'BearerAuth': []}],
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'username': {
                            'type': 'string',
                            'description': '用户名'
                        },
                        'nickname': {
                            'type': 'string',
                            'description': '昵称'
                        },
                        'gender': {
                            'type': 'integer',
                            'description': '性别'
                        },
                        'deptId': {
                            'type': 'integer',
                            'description': '部门ID'
                        },
                        'mobile': {
                            'type': 'string',
                            'description': '手机号'
                        },
                        'email': {
                            'type': 'string',
                            'description': '邮箱'
                        },
                        'roleIds': {
                            'type': 'array',
                            'items': {'type': 'integer'},
                            'description': '角色ID列表'
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
def create_user():
    """新增用户"""
    data = request.get_json()
    result = UserService.create_user_with_roles(data)
    
    if result.get('error'):
        return error(result['error'], 400)
    
    return success(result['data'], '新增成功')


@user_blueprint.route('/<int:user_id>/form', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['用户管理'],
    'summary': '获取用户表单数据',
    'description': '获取用户表单数据',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'user_id',
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
            'description': '用户不存在'
        }
    }
})
def get_user_form(user_id):
    """获取用户表单数据"""
    user_data = UserService.get_user_form_data(user_id)
    
    if not user_data:
        return error('用户不存在', 404)
    
    return success(user_data)


@user_blueprint.route('/<int:user_id>', methods=['PUT'])
@jwt_required
@swag_from({
    'tags': ['用户管理'],
    'summary': '更新用户',
    'description': '更新用户信息',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'user_id',
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
                        'username': {'type': 'string'},
                        'nickname': {'type': 'string'},
                        'gender': {'type': 'integer'},
                        'deptId': {'type': 'integer'},
                        'mobile': {'type': 'string'},
                        'email': {'type': 'string'},
                        'status': {'type': 'integer'},
                        'roleIds': {
                            'type': 'array',
                            'items': {'type': 'integer'}
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '更新成功'
        }
    }
})
def update_user(user_id):
    """更新用户信息"""
    data = request.get_json()
    result = UserService.update_user_with_roles(user_id, data)
    
    if result.get('error'):
        return error(result['error'], 400)
    
    return success(result.get('data'), '更新成功')


@user_blueprint.route('/<int:user_id>/status', methods=['PATCH'])
@jwt_required
@swag_from({
    'tags': ['用户管理'],
    'summary': '更新用户状态',
    'description': '更新用户状态（启用/禁用）',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'user_id',
            'in': 'path',
            'required': True,
            'schema': {'type': 'integer'}
        },
        {
            'name': 'status',
            'in': 'query',
            'required': True,
            'schema': {'type': 'integer', 'enum': [0, 1]}
        }
    ],
    'responses': {
        '200': {
            'description': '更新成功'
        }
    }
})
def update_user_status(user_id):
    """更新用户状态"""
    status = request.args.get('status', type=int)
    
    if status not in [0, 1]:
        return error('状态值只能为0或1', 400)
    
    result = UserService.update_user_status(user_id, status)
    
    if not result:
        return error('用户不存在', 404)
    
    return success(None, '更新成功')


@user_blueprint.route('/<int:user_id>/password', methods=['PUT'])
@jwt_required
@swag_from({
    'tags': ['用户管理'],
    'summary': '修改用户密码',
    'description': '修改用户密码',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'user_id',
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
                        'password': {
                            'type': 'string',
                            'description': '新密码'
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '修改成功'
        }
    }
})
def update_password(user_id):
    """修改用户密码"""
    data = request.get_json()
    password = data.get('password')
    
    if not password:
        return error('密码不能为空', 400)
    
    result = UserService.update_password(user_id, password)
    
    if not result:
        return error('用户不存在', 404)
    
    return success(None, '修改成功')


@user_blueprint.route('/<int:user_id>', methods=['DELETE'])
@jwt_required
@swag_from({
    'tags': ['用户管理'],
    'summary': '删除用户',
    'description': '删除用户（逻辑删除）',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'user_id',
            'in': 'path',
            'required': True,
            'schema': {'type': 'integer'}
        }
    ],
    'responses': {
        '200': {
            'description': '删除成功'
        }
    }
})
def delete_user(user_id):
    """删除用户"""
    result = UserService.delete_user(user_id)
    
    if not result:
        return error('用户不存在', 404)
    
    return success(None, '删除成功')