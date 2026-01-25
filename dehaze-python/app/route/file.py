"""
文件管理路由 - 使用 flask-openapi3 自动生成 Swagger 文档
"""
from io import BytesIO

from flask import request, current_app, send_file
from flask_openapi3 import APIBlueprint, Tag

from app.models import SysFile
from app.models.schema.file import FileIdQuery, FileCheckQuery, FileUploadResultVO
from app.service.file import upload_file_from_request
from app.utils.jwt_util import jwt_required
from app.utils.result import success, error


# 定义 Tag
file_tag = Tag(name="文件管理", description="文件相关接口")

# 创建 APIBlueprint（自动携带安全配置）
file_blueprint = APIBlueprint(
    "file",
    __name__,
    url_prefix="/api/v1/files",
    abp_tags=[file_tag],
    abp_security=[{"BearerAuth": []}]
)


@file_blueprint.post(
    "/",
    summary="文件上传",
    description="上传文件到存储服务"
)
@jwt_required
def upload_file_route():
    """文件上传"""
    if 'file' not in request.files:
        return error('请选择文件', 400)

    file = request.files['file']
    if file.filename == '':
        return error('请选择文件', 400)

    try:
        # 上传文件
        file_info = upload_file_from_request(file)
        return success({
            'id': file_info.id,
            'name': file_info.name,
            'url': file_info.url,
            'size': file_info.size,
            'md5': file_info.md5
        }, '文件上传成功')
    except Exception as e:
        return error(f'文件上传失败: {str(e)}', 500)


@file_blueprint.delete(
    "/",
    summary="文件删除",
    description="根据文件ID删除文件"
)
@jwt_required
def delete_file(query: FileIdQuery):
    """文件删除"""
    try:
        file_info = SysFile.query.get(query.fileId)
        if not file_info:
            return error('文件不存在', 404)

        # 从数据库删除文件记录
        from app.extensions import mysql
        mysql.session.delete(file_info)
        mysql.session.commit()

        return success(None, '文件删除成功')
    except Exception as e:
        from app.extensions import mysql
        mysql.session.rollback()
        return error(f'文件删除失败: {str(e)}', 500)


@file_blueprint.get(
    "/check",
    summary="文件校验",
    description="根据MD5值校验文件是否已存在"
)
@jwt_required
def check_file(query: FileCheckQuery):
    """文件校验"""
    file_info = SysFile.query.filter_by(md5=query.md5).first()
    exists = file_info is not None

    return success(exists)


@file_blueprint.get(
    "/download/<path:object_name>",
    summary="文件下载",
    description="根据对象名称下载文件"
)
@jwt_required
def download_file(object_name: str):
    """文件下载"""
    try:
        # 从MinIO获取文件
        minio_client = current_app.extensions["minio_client"]
        bucket_name = current_app.config["MINIO_BUCKET_NAME"]

        # 获取文件响应
        response = minio_client.get_object(bucket_name, object_name)

        # 获取文件名
        file_info = SysFile.query.filter_by(object_name=object_name).first()
        if not file_info:
            return error('文件不存在', 404)

        # 返回文件
        return send_file(
            BytesIO(response.read()),
            as_attachment=True,
            download_name=file_info.name
        )
    except Exception as e:
        return error(f'文件下载失败: {str(e)}', 500)
