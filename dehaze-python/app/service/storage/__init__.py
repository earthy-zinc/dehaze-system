"""
存储服务包

提供统一的存储抽象层，支持 MinIO、本地文件系统、nginx 静态服务三种后端。
URL 永远运行时拼接（baseUrl + object_name），不落库。
"""

from app.service.storage.base import StorageService
from app.service.storage.factory import (
    get_storage_by_name,
    get_storage_service,
    reset_storage_service,
)

__all__ = [
    "StorageService",
    "get_storage_by_name",
    "get_storage_service",
    "reset_storage_service",
]
