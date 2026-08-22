"""
MinIO 客户端单例与同步执行线程池

MinIO SDK 为同步客户端，业务调用需经线程池执行以避免阻塞事件循环。
"""

from concurrent.futures import ThreadPoolExecutor

from minio import Minio

from app.config import settings

# MinIO 操作线程池（MinIO SDK 是同步的，需要在线程池中执行）
minio_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="minio-ops")

# MinIO 客户端单例
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
