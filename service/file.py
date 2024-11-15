import os
import uuid
from datetime import datetime, timedelta
from http import HTTPMethod
from io import BytesIO

from werkzeug.datastructures import FileStorage

from algorithm import convert_size
from base import db
from base import base as app
from flask import current_app

from common.utils import calculate_bytes_md5
from model.SysFile import SysFile

bucket_name = current_app.config['MINIO_BUCKET_NAME']
minio_client = app.minio_client
def upload_file(file: FileStorage) -> SysFile:
    """
    上传文件到minio，并返回文件信息
    """
    # 获取文件名和扩展名
    filename = file.filename
    file_extension = filename.split('.')[-1]

    # 获取文件大小
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)

    file_bytes = BytesIO(file.read())

    # 根据md5从数据库中查询文件是否已存在
    file_md5 = calculate_bytes_md5(file_bytes)
    file_info = SysFile.query.filter_by(md5=file_md5).first()
    if file_info:
        return file_info

    # 文件不存在，上传到minio，然后数据库中新增一条数据
    object_name = datetime.now().strftime("%Y%m%d") + os.sep + uuid.uuid1().hex + "." + file_extension
    minio_client.put_object(bucket_name, object_name, file_bytes, file_size, content_type=file.content_type)
    file_url = minio_client.get_presigned_url(HTTPMethod.GET, bucket_name, object_name, expires=timedelta(days=365 * 30))

    file_info = SysFile(
        type=file_extension,
        url=file_url,
        name=filename,
        object_name=object_name,
        size=convert_size(file_size),
        path="",
        md5=file_md5,
        create_time=datetime.utcnow(),
        update_time=datetime.utcnow()
    )
    db.session.add(file_info)
    db.session.commit()
    return file_info

def read_file_from_url(url: str) -> BytesIO:
    file_info = SysFile.query.filter_by(url=url).first()
    assert file_info is not None, "文件不存在"
    file_response = minio_client.get_object(bucket_name, file_info.object_name)
    return BytesIO(file_response.read())


