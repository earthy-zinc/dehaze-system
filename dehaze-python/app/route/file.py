from io import BytesIO

from flask import Blueprint, request, jsonify, current_app, send_file

from app.models import SysFile
from app.route.utils import login_required
from app.service.file import upload_file_from_request
from app.utils.utils import result_util

file_blueprint = Blueprint('file', __name__, url_prefix='/api/v1/files')


@file_blueprint.route('/', methods=['POST'])
@login_required
def upload_file_route():
    """
    文件上传
    """
    if 'file' not in request.files:
        return jsonify(result_util(400, '请选择文件', None)), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(result_util(400, '请选择文件', None)), 400

    try:
        # 上传文件
        file_info = upload_file_from_request(file)
        return jsonify(result_util(200, '文件上传成功', {
            'id': file_info.id,
            'name': file_info.name,
            'url': file_info.url,
            'size': file_info.size,
            'md5': file_info.md5
        }))
    except Exception as e:
        return jsonify(result_util(500, f'文件上传失败: {str(e)}', None)), 500


@file_blueprint.route('/', methods=['DELETE'])
@login_required
def delete_file():
    """
    文件删除
    """
    file_id = request.args.get('fileId')
    if not file_id:
        return jsonify(result_util(400, '文件ID不能为空', None)), 400

    try:
        file_id = int(file_id)
        file_info = SysFile.query.get(file_id)
        if not file_info:
            return jsonify(result_util(404, '文件不存在', None)), 404

        # 从数据库删除文件记录
        from app.extensions import mysql
        mysql.session.delete(file_info)
        mysql.session.commit()

        return jsonify(result_util(200, '文件删除成功', None))
    except ValueError:
        return jsonify(result_util(400, '文件ID格式错误', None)), 400
    except Exception as e:
        from app.extensions import mysql
        mysql.session.rollback()
        return jsonify(result_util(500, f'文件删除失败: {str(e)}', None)), 500


@file_blueprint.route('/check', methods=['GET'])
@login_required
def check_file():
    """
    文件校验
    """
    md5 = request.args.get('md5')
    if not md5:
        return jsonify(result_util(400, '文件MD5不能为空', None)), 400

    file_info = SysFile.query.filter_by(md5=md5).first()
    exists = file_info is not None

    return jsonify(result_util(200, 'success', exists))


@file_blueprint.route('/download/<path:object_name>', methods=['GET'])
@login_required
def download_file(object_name):
    """
    文件下载
    """
    try:
        # 从MinIO获取文件
        minio_client = current_app.extensions["minio_client"]
        bucket_name = current_app.config["MINIO_BUCKET_NAME"]

        # 获取文件响应
        response = minio_client.get_object(bucket_name, object_name)

        # 获取文件名
        file_info = SysFile.query.filter_by(object_name=object_name).first()
        if not file_info:
            return jsonify(result_util(404, '文件不存在', None)), 404

        # 返回文件
        return send_file(
            BytesIO(response.read()),
            as_attachment=True,
            download_name=file_info.name
        )
    except Exception as e:
        return jsonify(result_util(500, f'文件下载失败: {str(e)}', None)), 500
