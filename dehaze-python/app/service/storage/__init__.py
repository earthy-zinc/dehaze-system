"""
存储服务包

提供统一的存储抽象层，支持 MinIO 和本地文件系统两种后端。
"""

from app.service.storage.base import StorageService
from app.service.storage.factory import get_storage_service, reset_storage_service

__all__ = ["StorageService", "get_storage_service", "reset_storage_service"]
