"""
存储服务工厂

提供两种取实例方式：
1. get_storage_service()：按全局默认配置 FILE_STORAGE_TYPE 取后端（上传/删除文件用）
2. get_storage_by_name(storage)：按 sys_file.storage 标识取后端（下载/读文件用）
"""

import logging

from app.config import settings
from app.service.storage.base import StorageService

logger = logging.getLogger(__name__)

# 存储服务单例缓存（按 storage 标识）
_storage_instances: dict[str, StorageService] = {}


def _create(storage: str) -> StorageService:
    """按 storage 标识创建存储服务实例"""
    storage = storage.lower()
    if storage == "minio":
        from app.service.storage.minio_storage import MinioStorageService

        return MinioStorageService()
    if storage == "local":
        from app.service.storage.local_storage import LocalStorageService

        return LocalStorageService()
    if storage == "nginx-static":
        from app.service.storage.nginx_storage import NginxStorageService

        return NginxStorageService()
    raise ValueError(f"不支持的存储类型: {storage}，可选: minio, local, nginx-static")


def get_storage_service() -> StorageService:
    """获取默认存储服务实例（单例，按 FILE_STORAGE_TYPE 配置）"""
    return get_storage_by_name(settings.FILE_STORAGE_TYPE)


def get_storage_by_name(storage: str) -> StorageService:
    """按 storage 标识获取存储服务实例（单例缓存）

    用于根据 sys_file.storage 字段选择对应后端进行下载/读取。
    """
    if storage in _storage_instances:
        return _storage_instances[storage]
    instance = _create(storage)
    _storage_instances[storage] = instance
    logger.info("存储服务初始化: %s", storage)
    return instance


def reset_storage_service() -> None:
    """重置存储服务实例缓存（主要用于测试）"""
    _storage_instances.clear()
