from flask import Blueprint, request, jsonify
from app.service.item_file_service import ItemFileService
from app.route.utils import login_required
from app.utils.utils import result_util

item_file_blueprint = Blueprint('item_file', __name__, url_prefix='/api/v1/dataset/image')


@item_file_blueprint.route('/', methods=['POST'])
@login_required
def upload_item_image():
    """
    上传数据项图片
    """
    if 'file' not in request.files:
        return jsonify(result_util(400, '请选择文件', None)), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(result_util(400, '请选择文件', None)), 400
    
    dataset_item_id = request.form.get('datasetItemId')
    if not dataset_item_id:
        return jsonify(result_util(400, '缺少参数datasetItemId', None)), 400
    
    type = request.form.get('type')
    if not type:
        return jsonify(result_util(400, '缺少参数type', None)), 400
    
    description = request.form.get('description', '')
    
    try:
        dataset_item_id = int(dataset_item_id)
        result = ItemFileService.save_item_file(dataset_item_id, file, type, description)
        
        if result['success']:
            return jsonify(result_util(200, '上传成功', result['data']))
        else:
            return jsonify(result_util(500, result['message'], None)), 500
    except ValueError:
        return jsonify(result_util(400, '参数格式错误', None)), 400
    except Exception as e:
        return jsonify(result_util(500, f'上传失败: {str(e)}', None)), 500


@item_file_blueprint.route('/', methods=['PUT'])
@login_required
def update_item_image():
    """
    修改数据项图片信息
    """
    item_file_id = request.args.get('itemFileId')
    type = request.args.get('type')
    
    if not item_file_id:
        return jsonify(result_util(400, '缺少参数itemFileId', None)), 400
    
    if not type:
        return jsonify(result_util(400, '缺少参数type', None)), 400
    
    # TODO: 实现修改数据项图片信息的逻辑
    return jsonify(result_util(500, '暂未实现', None)), 500


@item_file_blueprint.route('/', methods=['DELETE'])
@login_required
def delete_item_image():
    """
    删除数据项图片
    """
    item_file_id = request.args.get('itemFileId')
    if not item_file_id:
        return jsonify(result_util(400, '缺少参数itemFileId', None)), 400
    
    try:
        item_file_id = int(item_file_id)
        result = ItemFileService.delete_item_file(item_file_id)
        
        if result['success']:
            return jsonify(result_util(200, result['message'], None))
        else:
            return jsonify(result_util(500, result['message'], None)), 500
    except ValueError:
        return jsonify(result_util(400, '参数格式错误', None)), 400
    except Exception as e:
        return jsonify(result_util(500, f'删除失败: {str(e)}', None)), 500