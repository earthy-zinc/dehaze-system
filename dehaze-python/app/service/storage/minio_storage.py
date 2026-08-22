"""
MinIO 存储服务实现
"""

import logging
from collections.abc import Iterator
from io import BytesIO

from minio import Minio

from app.config import settings
from app.service.storage.base import StorageService

logger = logging.getLogger(__name__)


class MinioStorageService(StorageService):
    """
    MinIO 存储服务实现

    所有方法为同步操作，需在线程池中调用以避免阻塞事件循环。
    客户端实例在构造时创建，整个生命周期复用。
    """

    def __init__(self):
        self._client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

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
        except Exception:
            return False

    def get_size(self, bucket: str, object_name: str) -> int | None:
        try:
            stat = self._client.stat_object(bucket, object_name)
            return stat.size
        except Exception:
            return None

    def ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
            logger.info(f"已自动创建 MinIO Bucket: {bucket}")

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        objects = self._client.list_objects(bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects if obj.object_name is not None]
