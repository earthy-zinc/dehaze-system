"""
存储服务工厂

根据配置 FILE_STORAGE_TYPE 创建对应的存储服务实例。
"""

import logging
from typing import Optional

from app.config import settings
from app.service.storage.base import StorageService

logger = logging.getLogger(__name__)

# 存储服务单例缓存
_storage_instance: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """
    获取存储服务实例（单例）

    根据 FILE_STORAGE_TYPE 配置返回对应实现：
    - "minio": MinIO 对象存储
    - "local": 本地文件系统

    Returns:
        StorageService 实例
    """
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    storage_type = settings.FILE_STORAGE_TYPE.lower()

    if storage_type == "minio":
        from app.service.storage.minio_storage import MinioStorageService
        _storage_instance = MinioStorageService()
        logger.info("存储服务初始化: MinIO")
    elif storage_type == "local":
        from app.service.storage.local_storage import LocalStorageService
        _storage_instance = LocalStorageService()
        logger.info(f"存储服务初始化: 本地存储 ({settings.LOCAL_STORAGE_PATH})")
    else:
        raise ValueError(f"不支持的存储类型: {storage_type}，可选: minio, local")

    return _storage_instance


def reset_storage_service() -> None:
    """重置存储服务实例（主要用于测试）"""
    global _storage_instance
    _storage_instance = None
