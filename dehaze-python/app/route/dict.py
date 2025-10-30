from flasgger import swag_from
from flask import Blueprint, request

from app.service.dict_service import DictService, DictTypeService
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error

dict_blueprint = Blueprint('dict', __name__, url_prefix='/api/v1/dict')


@dict_blueprint.route('/page', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['字典管理'],
    'summary': '字典分页列表',
    'description': '获取字典分页列表',
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
        },
        {
            'name': 'typeCode',
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
def get_dict_page():
    """字典分页列表"""
    page = request.args.get('pageNum', 1, type=int)
    page_size = request.args.get('pageSize', 10, type=int)
    keywords = request.args.get('keywords', type=str)
    type_code = request.args.get('typeCode', type=str)

    dict_items, total = DictService.get_dict_page(page, page_size, keywords, type_code)

    dict_list = []
    for item in dict_items:
        dict_list.append({
            'id': item.id,
            'typeCode': item.type_code,
            'name': item.name,
            'value': item.value,
            'status': item.status,
            'sort': item.sort
        })

    return success({
        'list': dict_list,
        'total': total,
        'pageNum': page,
        'pageSize': page_size
    })


@dict_blueprint.route('/<int:dict_id>/form', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['字典管理'],
    'summary': '字典表单数据',
    'description': '获取字典表单数据',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'dict_id',
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
            'description': '字典不存在'
        }
    }
})
def get_dict_form(dict_id):
    """获取字典表单数据"""
    dict_data = DictService.get_dict_form(dict_id)

    if not dict_data:
        return error('字典不存在', 404)

    return success(dict_data)


@dict_blueprint.route('/', methods=['POST'])
@jwt_required
@swag_from({
    'tags': ['字典管理'],
    'summary': '新增字典',
    'description': '新增字典项',
    'security': [{'BearerAuth': []}],
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'typeCode': {
                            'type': 'string',
                            'description': '字典类型编码'
                        },
                        'name': {
                            'type': 'string',
                            'description': '字典项名称'
                        },
                        'value': {
                            'type': 'string',
                            'description': '字典项值'
                        },
                        'status': {
                            'type': 'integer',
                            'description': '状态(1:正常;0:禁用)'
                        },
                        'sort': {
                            'type': 'integer',
                            'description': '排序'
                        },
                        'remark': {
                            'type': 'string',
                            'description': '备注'
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
def create_dict():
    """新增字典"""
    data = request.get_json()
    result = DictService.create_dict(data)

    if result:
        return success(None, '新增成功')
    else:
        return error('新增失败', 400)


@dict_blueprint.route('/<int:dict_id>', methods=['PUT'])
@jwt_required
@swag_from({
    'tags': ['字典管理'],
    'summary': '修改字典',
    'description': '修改字典项',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'dict_id',
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
                        'typeCode': {
                            'type': 'string',
                            'description': '字典类型编码'
                        },
                        'name': {
                            'type': 'string',
                            'description': '字典项名称'
                        },
                        'value': {
                            'type': 'string',
                            'description': '字典项值'
                        },
                        'status': {
                            'type': 'integer',
                            'description': '状态(1:正常;0:禁用)'
                        },
                        'sort': {
                            'type': 'integer',
                            'description': '排序'
                        },
                        'remark': {
                            'type': 'string',
                            'description': '备注'
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '修改成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def update_dict(dict_id):
    """修改字典"""
    data = request.get_json()
    result = DictService.update_dict(dict_id, data)

    if result:
        return success(None, '修改成功')
    else:
        return error('修改失败', 400)


@dict_blueprint.route('/<string:dict_ids>', methods=['DELETE'])
@jwt_required
@swag_from({
    'tags': ['字典管理'],
    'summary': '删除字典',
    'description': '删除字典项',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'dict_ids',
            'in': 'path',
            'required': True,
            'schema': {'type': 'string'},
            'description': '字典ID，多个以英文逗号(,)分割'
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
def delete_dict(dict_ids):
    """删除字典"""
    try:
        id_list = [int(id) for id in dict_ids.split(',')]
        result = DictService.delete_dict(id_list)

        if result:
            return success(None, '删除成功')
        else:
            return error('删除失败', 400)
    except Exception as e:
        return error('参数错误', 400)


@dict_blueprint.route('/<string:type_code>/options', methods=['GET'])
@swag_from({
    'tags': ['字典管理'],
    'summary': '字典下拉列表',
    'description': '获取字典下拉列表',
    'parameters': [
        {
            'name': 'type_code',
            'in': 'path',
            'required': True,
            'schema': {'type': 'string'}
        }
    ],
    'responses': {
        '200': {
            'description': '获取成功'
        }
    }
})
def list_dict_options(type_code):
    """字典下拉列表"""
    options = DictService.list_dict_options(type_code)
    return success(options)


# 字典类型相关接口
@dict_blueprint.route('/types/page', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['字典类型管理'],
    'summary': '字典类型分页列表',
    'description': '获取字典类型分页列表',
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
def get_dict_type_page():
    """字典类型分页列表"""
    page = request.args.get('pageNum', 1, type=int)
    page_size = request.args.get('pageSize', 10, type=int)
    keywords = request.args.get('keywords', type=str)

    dict_types, total = DictTypeService.get_dict_type_page(page, page_size, keywords)

    type_list = []
    for item in dict_types:
        type_list.append({
            'id': item.id,
            'name': item.name,
            'code': item.code,
            'status': item.status,
            'remark': item.remark
        })

    return success({
        'list': type_list,
        'total': total,
        'pageNum': page,
        'pageSize': page_size
    })


@dict_blueprint.route('/types/<int:type_id>/form', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['字典类型管理'],
    'summary': '字典类型表单数据',
    'description': '获取字典类型表单数据',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'type_id',
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
            'description': '字典类型不存在'
        }
    }
})
def get_dict_type_form(type_id):
    """获取字典类型表单数据"""
    dict_type_data = DictTypeService.get_dict_type_form(type_id)

    if not dict_type_data:
        return error('字典类型不存在', 404)

    return success(dict_type_data)


@dict_blueprint.route('/types', methods=['POST'])
@jwt_required
@swag_from({
    'tags': ['字典类型管理'],
    'summary': '新增字典类型',
    'description': '新增字典类型',
    'security': [{'BearerAuth': []}],
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': '类型名称'
                        },
                        'code': {
                            'type': 'string',
                            'description': '类型编码'
                        },
                        'status': {
                            'type': 'integer',
                            'description': '状态(1:正常;0:禁用)'
                        },
                        'remark': {
                            'type': 'string',
                            'description': '备注'
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
def create_dict_type():
    """新增字典类型"""
    data = request.get_json()
    result = DictTypeService.create_dict_type(data)

    if result:
        return success(None, '新增成功')
    else:
        return error('新增失败', 400)


@dict_blueprint.route('/types/<int:type_id>', methods=['PUT'])
@jwt_required
@swag_from({
    'tags': ['字典类型管理'],
    'summary': '修改字典类型',
    'description': '修改字典类型',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'type_id',
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
                            'description': '类型名称'
                        },
                        'code': {
                            'type': 'string',
                            'description': '类型编码'
                        },
                        'status': {
                            'type': 'integer',
                            'description': '状态(1:正常;0:禁用)'
                        },
                        'remark': {
                            'type': 'string',
                            'description': '备注'
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '修改成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def update_dict_type(type_id):
    """修改字典类型"""
    data = request.get_json()
    result = DictTypeService.update_dict_type(type_id, data)

    if result:
        return success(None, '修改成功')
    else:
        return error('修改失败', 400)


@dict_blueprint.route('/types/<string:type_ids>', methods=['DELETE'])
@jwt_required
@swag_from({
    'tags': ['字典类型管理'],
    'summary': '删除字典类型',
    'description': '删除字典类型',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'type_ids',
            'in': 'path',
            'required': True,
            'schema': {'type': 'string'},
            'description': '字典类型ID，多个以英文逗号(,)分割'
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
def delete_dict_types(type_ids):
    """删除字典类型"""
    try:
        id_list = [int(id) for id in type_ids.split(',')]
        result = DictTypeService.delete_dict_types(id_list)

        if result:
            return success(None, '删除成功')
        else:
            return error('删除失败', 400)
    except Exception as e:
        return error('参数错误', 400)
