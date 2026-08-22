"""
存储服务包

提供统一的存储抽象层，支持 MinIO、本地文件系统、nginx 静态服务三种后端。
URL 永远运行时拼接（baseUrl + object_name），不落库。

显式导入（不做包级 re-export）：
    from app.service.storage.factory import get_storage_service
"""
