"""
MinIO 存储服务实现
"""

import logging
from collections.abc import Iterator
from io import BytesIO

from minio import Minio
from minio.error import S3Error

from app.config import settings
from app.infrastructure.storage.minio_client import get_minio_client
from app.service.storage.base import StorageService

logger = logging.getLogger(__name__)

# stat_object 的"对象不存在"错误码：存在性/大小探测的正常结果，不作为故障上报
_NOT_FOUND_CODES = frozenset({"NoSuchKey", "NoSuchBucket"})


class MinioStorageService(StorageService):
    """
    MinIO 存储服务实现

    所有方法为同步操作，需在线程池中调用以避免阻塞事件循环。
    客户端复用基础设施层的统一单例。
    """

    def __init__(self):
        self._client = get_minio_client()

    @property
    def name(self) -> str:
        return "minio"

    @property
    def base_url(self) -> str:
        return settings.FILE_STORAGE_BASE_URLS["minio"]

    @property
    def client(self) -> Minio:
        return self._client

    def upload(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str,
    ) -> None:
        self.ensure_bucket(bucket)
        self._client.put_object(
            bucket,
            object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def download(self, bucket: str, object_name: str) -> bytes:
        response = self._client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def download_stream(
        self, bucket: str, object_name: str, chunk_size: int = 1024 * 1024
    ) -> Iterator[bytes]:
        response = self._client.get_object(bucket, object_name)
        try:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            response.close()
            response.release_conn()

    def delete(self, bucket: str, object_name: str) -> None:
        self._client.remove_object(bucket, object_name)

    def exists(self, bucket: str, object_name: str) -> bool:
        try:
            self._client.stat_object(bucket, object_name)
            return True
        except S3Error as e:
            _log_stat_error(bucket, object_name, e)
            return False

    def get_size(self, bucket: str, object_name: str) -> int | None:
        try:
            stat = self._client.stat_object(bucket, object_name)
            return stat.size
        except S3Error as e:
            _log_stat_error(bucket, object_name, e)
            return None

    def ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
            logger.info(f"已自动创建 MinIO Bucket: {bucket}")

    def list_objects(self, bucket: str, prefix: str = "") -> list[tuple[str, float]]:
        objects = self._client.list_objects(bucket, prefix=prefix, recursive=True)
        return [
            (obj.object_name, obj.last_modified.timestamp() if obj.last_modified else 0.0)
            for obj in objects
            if obj.object_name is not None
        ]


def _log_stat_error(bucket: str, object_name: str, err: S3Error) -> None:
    """stat_object 失败留痕：对象不存在（NoSuchKey/NoSuchBucket）是探测的正常结果，
    其余 S3 错误（鉴权/存储侧故障）必须可见，不得与"不存在"混为一谈"""
    if err.code not in _NOT_FOUND_CODES:
        logger.warning("MinIO stat_object 失败 [bucket=%s object=%s]: %s", bucket, object_name, err)
