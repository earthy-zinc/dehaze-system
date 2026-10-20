"""
MinIO 客户端单例

MinIO SDK 为同步客户端，业务调用统一经 app.service.storage.executor 的
storage_executor 线程池执行以避免阻塞事件循环。
"""

from minio import Minio

from app.config import settings

_minio_client: Minio | None = None


def get_minio_client() -> Minio:
    """获取 MinIO 客户端单例实例"""
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
    return _minio_client
