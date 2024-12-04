import os
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Tuple
from urllib.parse import urlparse

import requests
from flask import current_app
from werkzeug.datastructures import FileStorage

from app.models import SysFile, SysWpxFile
from app.utils.error import BusinessException
from app.utils.file import calculate_bytes_md5, convert_size, get_file_bytes
from app.extensions import mysql


def upload_file_from_request(file: FileStorage) -> SysFile:
    """
    上传文件到 MinIO 并返回文件信息
    :param file: FileStorage 对象
    :return: SysFile 数据库记录
    """
    file_bytes = BytesIO(file.read())
    return _upload_to_storage(
        filename=file.filename,
        content_type=file.mimetype,
        file_bytes=file_bytes,
        file_size=file.content_length
    )


def upload_file(filename: str, content_type: str, file_bytes: BytesIO) -> SysFile:
    """
    上传文件到 MinIO 并返回文件信息
    :param filename: 文件名
    :param content_type: 文件类型
    :param file_bytes: 文件的字节流
    :return: SysFile 数据库记录
    """
    file_size = len(file_bytes.getvalue())
    return _upload_to_storage(filename, content_type, file_bytes, file_size)


def read_file_from_url(url: str, flag: bool=False) -> tuple[BytesIO, SysFile]:
    """
    从 URL 中读取文件
    :param url: 文件的 URL
    :param flag:
    :return: 文件内容 BytesIO 对象
    """
    bucket_name = current_app.config["MINIO_BUCKET_NAME"]
    minio_client = current_app.extensions["minio_client"]

    # 检查文件是否已存储
    file_info = SysFile.query.filter_by(url=url).first()
    if not file_info:
        # 文件未存储，下载后上传
        filename, content_type, file_bytes = _fetch_file_from_url(url)
        file_info = _upload_to_storage(
            filename=filename, content_type=content_type, file_bytes=file_bytes, file_size=len(file_bytes.getvalue())
        )
    # 判断当前文件是否应需要转换
    if flag:
        file_info = _get_new_file_info(file_info)
    # 从 MinIO 获取文件内容
    file_response = minio_client.get_object(bucket_name, file_info.object_name)
    return BytesIO(file_response.read()), file_info

def _get_new_file_info(old_file_info: SysFile) -> SysFile:
    """
    根据旧文件的md5，从SysWpxFile数据库中获取新文件的md5, path
    如果能找到新文件md5对应的SysFile，则返回SysFile
    否则根据 path 中加载数据然后上传到 minio 中

    :param old_file_info:
    :return: new_file_info
    """
    old_md5 = old_file_info.md5
    sys_wpx_file: SysWpxFile = SysWpxFile.query.filter_by(origin_md5=old_md5).first()

    if not sys_wpx_file: raise BusinessException("未找到对应的转换文件")

    new_md5 = sys_wpx_file.new_md5

    new_file_info = SysFile.query.filter_by(md5=new_md5).first()

    if new_file_info: return new_file_info

    dataset_path = current_app.config.get("DATASET_PATH", "")
    new_path = sys_wpx_file.new_path
    file_path = os.path.join(dataset_path, new_path)

    if not os.path.exists(file_path): raise BusinessException("文件不存在")

    file_name = os.path.basename(file_path)
    file_bytes = get_file_bytes(file_path)
    new_file_info = upload_file(file_name, "image/png", file_bytes)
    return new_file_info


def _fetch_file_from_url(url: str) -> Tuple[str, str, BytesIO]:
    """
    从 URL 获取文件并转换为 BytesIO
    :param url: 文件 URL
    :return: (文件名, Content-Type, 文件字节流)
    """
    try:
        parsed_url = urlparse(url)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "unknown")
        filename = os.path.basename(parsed_url.path)
        return filename, content_type, BytesIO(response.content)
    except ValueError as e:
        raise BusinessException("URL 格式错误", e)
    except requests.exceptions.RequestException as e:
        raise BusinessException("请求图片时出错", e)


def _upload_to_storage(
        filename: str, content_type: str, file_bytes: BytesIO, file_size: int
) -> SysFile:
    """
    上传文件到 MinIO 并保存文件元数据到数据库
    :param filename: 文件名
    :param content_type: 文件类型
    :param file_bytes: 文件字节流
    :param file_size: 文件大小（字节）
    :return: SysFile 数据库记录
    """
    bucket_name = current_app.config["MINIO_BUCKET_NAME"]
    minio_client = current_app.extensions["minio_client"]
    # 检查文件是否已存在
    file_md5 = calculate_bytes_md5(file_bytes)
    existing_file = SysFile.query.filter_by(md5=file_md5).first()
    if existing_file:
        return existing_file

    # 上传文件到 MinIO
    file_extension = filename.rsplit(".", 1)[-1]
    object_name = _generate_object_name(file_extension)
    minio_client.put_object(bucket_name, object_name, file_bytes, file_size, content_type=content_type)

    # 生成文件访问 URL
    file_url = minio_client.get_presigned_url(
        "GET", bucket_name, object_name, expires=timedelta(days=7)
    )

    # 保存文件信息到数据库
    new_file = SysFile(
        type=file_extension,
        url=file_url,
        name=filename,
        object_name=object_name,
        size=convert_size(file_size),
        path="",
        md5=file_md5,
    )
    mysql.session.add(new_file)
    mysql.session.commit()
    return new_file


def _generate_object_name(extension: str) -> str:
    """
    生成唯一的对象名
    :param extension: 文件扩展名
    :return: 对象名
    """
    return f"{datetime.now().strftime('%Y%m%d')}/{uuid.uuid4().hex}.{extension}"
