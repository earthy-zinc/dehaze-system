from flasgger import swag_from
from flask import Blueprint, request

from app.service.dataset_service import DatasetService, DatasetItemService
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error

dataset_blueprint = Blueprint('dataset', __name__, url_prefix='/api/v1/dataset')


@dataset_blueprint.route('/', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['数据集管理'],
    'summary': '数据集列表',
    'description': '获取数据集列表（树形结构）',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'keywords',
            'in': 'query',
            'required': False,
            'schema': {'type': 'string'},
            'description': '关键字（数据集名称）'
        }
    ],
    'responses': {
        '200': {
            'description': '获取成功'
        }
    }
})
def list_datasets():
    """获取数据集列表"""
    keywords = request.args.get('keywords', type=str)

    dataset_list = DatasetService.get_dataset_list(keywords)
    return success(dataset_list)


@dataset_blueprint.route('/options', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['数据集管理'],
    'summary': '数据集下拉列表',
    'description': '获取数据集下拉列表',
    'security': [{'BearerAuth': []}],
    'responses': {
        '200': {
            'description': '获取成功'
        }
    }
})
def list_dataset_options():
    """数据集下拉列表"""
    options = DatasetService.get_dataset_options()
    return success(options)


@dataset_blueprint.route('/<int:dataset_id>', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['数据集管理'],
    'summary': '数据集信息',
    'description': '根据ID获取数据集信息',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'dataset_id',
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
            'description': '数据集不存在'
        }
    }
})
def get_dataset_info(dataset_id):
    """获取数据集信息"""
    dataset = DatasetService.get_dataset_by_id(dataset_id)

    if dataset is None:
        return error('数据集不存在', 404)

    return success(dataset)


@dataset_blueprint.route('/', methods=['POST'])
@jwt_required
@swag_from({
    'tags': ['数据集管理'],
    'summary': '新增数据集',
    'description': '新增数据集',
    'security': [{'BearerAuth': []}],
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'parentId': {'type': 'integer', 'description': '父数据集ID'},
                        'type': {'type': 'string', 'description': '数据集类型'},
                        'name': {'type': 'string', 'description': '数据集名称'},
                        'description': {'type': 'string', 'description': '数据集描述'},
                        'path': {'type': 'string', 'description': '存储位置'},
                        'status': {'type': 'integer', 'description': '状态(1:启用；0:禁用)'}
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
def add_dataset():
    """新增数据集"""
    data = request.get_json()
    result = DatasetService.create_dataset(data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '保存成功')


@dataset_blueprint.route('/<int:dataset_id>', methods=['PUT'])
@jwt_required
@swag_from({
    'tags': ['数据集管理'],
    'summary': '修改数据集',
    'description': '修改数据集',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'dataset_id',
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
                        'parentId': {'type': 'integer', 'description': '父数据集ID'},
                        'type': {'type': 'string', 'description': '数据集类型'},
                        'name': {'type': 'string', 'description': '数据集名称'},
                        'description': {'type': 'string', 'description': '数据集描述'},
                        'path': {'type': 'string', 'description': '存储位置'},
                        'status': {'type': 'integer', 'description': '状态(1:启用；0:禁用)'}
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
def update_dataset(dataset_id):
    """修改数据集"""
    data = request.get_json()
    result = DatasetService.update_dataset(dataset_id, data)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '保存成功')


@dataset_blueprint.route('/', methods=['DELETE'])
@jwt_required
@swag_from({
    'tags': ['数据集管理'],
    'summary': '删除数据集',
    'description': '删除数据集',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'ids',
            'in': 'query',
            'required': True,
            'schema': {'type': 'array', 'items': {'type': 'integer'}},
            'description': '数据集ID列表',
            'style': 'form',
            'explode': False
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
def delete_datasets():
    """删除数据集"""
    ids = request.args.get('ids', type=str)
    if not ids:
        return error('参数错误', 400)

    try:
        dataset_ids = [int(id) for id in ids.split(',')]
    except ValueError:
        return error('参数格式错误', 400)

    result = DatasetService.delete_datasets(dataset_ids)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '删除成功')


@dataset_blueprint.route('/<int:dataset_id>/images', methods=['GET'])
@jwt_required
@swag_from({
    'tags': ['数据集管理'],
    'summary': '数据集图片',
    'description': '获取数据集图片项（分页）',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'dataset_id',
            'in': 'path',
            'required': True,
            'schema': {'type': 'integer'}
        },
        {
            'name': 'pageNum',
            'in': 'query',
            'required': False,
            'schema': {'type': 'integer', 'default': 1},
            'description': '页码'
        },
        {
            'name': 'pageSize',
            'in': 'query',
            'required': False,
            'schema': {'type': 'integer', 'default': 10},
            'description': '每页数量'
        }
    ],
    'responses': {
        '200': {
            'description': '获取成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def get_dataset_images(dataset_id):
    """获取数据集图片项"""
    page_num = request.args.get('pageNum', 1, type=int)
    page_size = request.args.get('pageSize', 10, type=int)

    result = DatasetService.get_image_items(dataset_id, page_num, page_size)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'])


# 数据集项相关接口
@dataset_blueprint.route('/item', methods=['POST'])
@jwt_required
@swag_from({
    'tags': ['数据集项管理'],
    'summary': '新增数据项',
    'description': '新增数据项',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'datasetId',
            'in': 'query',
            'required': True,
            'schema': {'type': 'integer'},
            'description': '所属数据集ID'
        },
        {
            'name': 'name',
            'in': 'query',
            'required': False,
            'schema': {'type': 'string'},
            'description': '数据项名称'
        }
    ],
    'responses': {
        '200': {
            'description': '创建成功'
        },
        '400': {
            'description': '参数错误'
        }
    }
})
def add_dataset_item():
    """新增数据项"""
    dataset_id = request.args.get('datasetId', type=int)
    name = request.args.get('name', type=str)

    if not dataset_id:
        return error('缺少参数datasetId', 400)

    result = DatasetItemService.create_dataset_item(dataset_id, name)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '创建成功')


@dataset_blueprint.route('/item', methods=['PUT'])
@jwt_required
@swag_from({
    'tags': ['数据集项管理'],
    'summary': '修改数据项',
    'description': '修改数据项',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'datasetItemId',
            'in': 'query',
            'required': True,
            'schema': {'type': 'integer'},
            'description': '数据项ID'
        },
        {
            'name': 'name',
            'in': 'query',
            'required': False,
            'schema': {'type': 'string'},
            'description': '数据项名称'
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
def update_dataset_item():
    """修改数据项"""
    dataset_item_id = request.args.get('datasetItemId', type=int)
    name = request.args.get('name', type=str)

    if not dataset_item_id:
        return error('缺少参数datasetItemId', 400)

    if not name:
        return error('缺少参数name', 400)

    result = DatasetItemService.update_dataset_item(dataset_item_id, name)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '更新成功')


@dataset_blueprint.route('/item', methods=['DELETE'])
@jwt_required
@swag_from({
    'tags': ['数据集项管理'],
    'summary': '删除数据项',
    'description': '删除数据项',
    'security': [{'BearerAuth': []}],
    'parameters': [
        {
            'name': 'datasetItemId',
            'in': 'query',
            'required': True,
            'schema': {'type': 'integer'},
            'description': '数据项ID'
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
def delete_dataset_item():
    """删除数据项"""
    dataset_item_id = request.args.get('datasetItemId', type=int)

    if not dataset_item_id:
        return error('缺少参数datasetItemId', 400)

    result = DatasetItemService.delete_dataset_item(dataset_item_id)

    if 'error' in result:
        return error(result['error'], 400)

    return success(result['data'], '删除成功')
