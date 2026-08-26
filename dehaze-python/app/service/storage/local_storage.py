"""
本地文件存储服务实现
"""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

from app.config import settings
from app.service.storage.base import StorageService

logger = logging.getLogger(__name__)


class LocalStorageService(StorageService):
    """
    本地文件系统存储实现

    文件存储在 LOCAL_STORAGE_PATH 指定目录下。
    bucket 映射为子目录，object_name 保持其路径结构。
    适用于开发环境或无 MinIO 的部署场景。
    """

    def __init__(self, base_dir: str | None = None):
        # base_dir 允许测试注入 tmp_path 做真实文件读写，生产默认取配置目录
        self._base_path = Path(base_dir) if base_dir else Path(settings.LOCAL_STORAGE_PATH)

    @property
    def name(self) -> str:
        return "local"

    @property
    def base_url(self) -> str:
        return settings.FILE_STORAGE_BASE_URLS["local"]

    def _resolve_path(self, bucket: str, object_name: str) -> Path:
        """解析完整文件路径"""
        full_path = self._base_path / bucket / object_name
        # 安全检查：确保路径不会逃逸出基础目录
        resolved = full_path.resolve()
        base_resolved = self._base_path.resolve()
        if not str(resolved).startswith(str(base_resolved)):
            raise ValueError(f"路径安全检查失败: {object_name}")
        return full_path

    def upload(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str,
    ) -> None:
        file_path = self._resolve_path(bucket, object_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)

    def download(self, bucket: str, object_name: str) -> bytes:
        file_path = self._resolve_path(bucket, object_name)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {object_name}")
        return file_path.read_bytes()

    def download_stream(
        self, bucket: str, object_name: str, chunk_size: int = 1024 * 1024
    ) -> Iterator[bytes]:
        file_path = self._resolve_path(bucket, object_name)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {object_name}")
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def delete(self, bucket: str, object_name: str) -> None:
        file_path = self._resolve_path(bucket, object_name)
        if file_path.exists():
            file_path.unlink()

    def exists(self, bucket: str, object_name: str) -> bool:
        file_path = self._resolve_path(bucket, object_name)
        return file_path.exists()

    def get_size(self, bucket: str, object_name: str) -> int | None:
        file_path = self._resolve_path(bucket, object_name)
        if file_path.exists():
            return file_path.stat().st_size
        return None

    def ensure_bucket(self, bucket: str) -> None:
        bucket_path = self._base_path / bucket
        bucket_path.mkdir(parents=True, exist_ok=True)

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        bucket_path = self._base_path / bucket
        if not bucket_path.exists():
            return []

        prefix_path = bucket_path / prefix if prefix else bucket_path
        if not prefix_path.exists():
            return []

        results = []
        for root, _, files in os.walk(prefix_path):
            for f in files:
                full = Path(root) / f
                relative = full.relative_to(bucket_path)
                results.append(str(relative))
        return results
