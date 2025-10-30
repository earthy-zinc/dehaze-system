from flask import Blueprint, request, jsonify

from app.route.utils import login_required, request_params_verify
from app.service.algorithm_service import AlgorithmService
from app.utils.utils import result_util

algorithm_blueprint = Blueprint('algorithm', __name__, url_prefix='/api/v1/algorithm')


@algorithm_blueprint.route('/', methods=['GET'])
@login_required
def list_algorithms():
    """
    获取算法树形表格
    """
    keywords = request.args.get('keywords', None)
    algorithms = AlgorithmService.get_algorithm_list(keywords)
    return jsonify(result_util(200, 'success', algorithms))


@algorithm_blueprint.route('/options', methods=['GET'])
@login_required
def get_algorithm_options():
    """
    获取模型下拉选项列表
    """
    options = AlgorithmService.get_algorithm_options()
    return jsonify(result_util(200, 'success', options))


@algorithm_blueprint.route('/<int:algorithm_id>', methods=['GET'])
@login_required
def get_algorithm_by_id(algorithm_id):
    """
    根据ID获取算法信息
    """
    algorithm = AlgorithmService.get_algorithm_by_id(algorithm_id)
    if algorithm:
        return jsonify(result_util(200, 'success', algorithm))
    else:
        return jsonify(result_util(404, '算法不存在', None))


@algorithm_blueprint.route('/', methods=['POST'])
@login_required
@request_params_verify(['name'])
def add_algorithm():
    """
    新增算法
    """
    data = request.get_json()
    result = AlgorithmService.create_algorithm(data)
    if result['success']:
        return jsonify(result_util(200, result['message'], None))
    else:
        return jsonify(result_util(500, result['message'], None)), 500


@algorithm_blueprint.route('/<int:algorithm_id>', methods=['PUT'])
@login_required
@request_params_verify(['name'])
def update_algorithm(algorithm_id):
    """
    修改算法
    """
    data = request.get_json()
    result = AlgorithmService.update_algorithm(algorithm_id, data)
    if result['success']:
        return jsonify(result_util(200, result['message'], None))
    else:
        if '不存在' in result['message']:
            return jsonify(result_util(404, result['message'], None)), 404
        else:
            return jsonify(result_util(500, result['message'], None)), 500


@algorithm_blueprint.route('/', methods=['DELETE'])
@login_required
def delete_algorithms():
    """
    删除算法
    """
    ids = request.args.get('ids')
    if not ids:
        return jsonify(result_util(400, '请选择要删除的算法', None)), 400

    try:
        algorithm_ids = [int(i) for i in ids.split(',')]
        result = AlgorithmService.delete_algorithms(algorithm_ids)
        if result['success']:
            return jsonify(result_util(200, result['message'], None))
        else:
            return jsonify(result_util(500, result['message'], None)), 500
    except ValueError:
        return jsonify(result_util(400, '参数格式错误', None)), 400
