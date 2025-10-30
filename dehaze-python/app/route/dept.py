from flask import Blueprint, request, jsonify

from app.route.utils import login_required
from app.service.dept_service import DeptService
from app.utils.utils import result_util

dept_blueprint = Blueprint('dept', __name__, url_prefix='/api/v1/dept')


@dept_blueprint.route('/', methods=['GET'])
@login_required
def list_depts():
    """
    获取部门列表
    """
    keywords = request.args.get('keywords', None)
    status = request.args.get('status', None)
    if status is not None:
        status = int(status)

    depts = DeptService.get_dept_list(keywords, status)
    return jsonify(result_util(200, 'success', depts))


@dept_blueprint.route('/options', methods=['GET'])
@login_required
def list_dept_options():
    """
    获取部门下拉选项
    """
    options = DeptService.get_dept_options()
    return jsonify(result_util(200, 'success', options))


@dept_blueprint.route('/<int:dept_id>/form', methods=['GET'])
@login_required
def get_dept_form(dept_id):
    """
    获取部门表单数据
    """
    dept_form = DeptService.get_dept_form(dept_id)
    if dept_form:
        return jsonify(result_util(200, 'success', dept_form))
    else:
        return jsonify(result_util(404, '部门不存在', None)), 404


@dept_blueprint.route('/', methods=['POST'])
@login_required
def add_dept():
    """
    新增部门
    """
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify(result_util(400, '部门名称不能为空', None)), 400

    if 'parent_id' not in data:
        return jsonify(result_util(400, '父部门ID不能为空', None)), 400

    result = DeptService.create_dept(data)
    if result['success']:
        return jsonify(result_util(200, result['message'], result['data']))
    else:
        return jsonify(result_util(500, result['message'], None)), 500


@dept_blueprint.route('/<int:dept_id>', methods=['PUT'])
@login_required
def update_dept(dept_id):
    """
    修改部门
    """
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify(result_util(400, '部门名称不能为空', None)), 400

    result = DeptService.update_dept(dept_id, data)
    if result['success']:
        return jsonify(result_util(200, result['message'], result['data']))
    else:
        if '不存在' in result['message']:
            return jsonify(result_util(404, result['message'], None)), 404
        else:
            return jsonify(result_util(500, result['message'], None)), 500


@dept_blueprint.route('/<string:ids>', methods=['DELETE'])
@login_required
def delete_depts(ids):
    """
    删除部门
    """
    try:
        dept_ids = [int(i) for i in ids.split(',')]
        result = DeptService.delete_depts(dept_ids)
        if result['success']:
            return jsonify(result_util(200, result['message'], None))
        else:
            return jsonify(result_util(500, result['message'], None)), 500
    except ValueError:
        return jsonify(result_util(400, '参数格式错误', None)), 400
