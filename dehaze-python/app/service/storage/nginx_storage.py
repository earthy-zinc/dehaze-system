"""
nginx 静态服务存储后端实现

用于 nginx 直服的静态资源（数据集文件 datasets/...、模型权重 models/...）。
文件已存在于 nginx 静态目录，本后端只提供 download/get_url/exists 等读取操作，
不支持 upload（数据集文件由外部脚本/部署写入 nginx 目录）。
"""

import logging
from collections.abc import Iterator

import requests

from app.config import settings
from app.service.storage.base import StorageService

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024


class NginxStorageService(StorageService):
    """
    nginx 静态服务后端

    baseUrl 为 nginx 静态服务根地址（如 http://host:9000），不带任何资源子路径。
    object_name 自带资源前缀（datasets/...、models/...），URL 拼接为：
        {baseUrl}/{object_name}  →  http://host:9000/datasets/AECR-Net/clear/01.jpg
    """

    @property
    def name(self) -> str:
        return "nginx-static"

    @property
    def base_url(self) -> str:
        return settings.FILE_STORAGE_BASE_URLS["nginx-static"]

    def _full_url(self, object_name: str) -> str:
        return f"{self.base_url.rstrip('/')}/{object_name}"

    def upload(self, bucket: str, object_name: str, data: bytes, content_type: str) -> None:
        # nginx-static 后端不管理上传，文件由外部部署/脚本写入
        raise NotImplementedError("nginx-static 后端不支持上传，文件已存在于 nginx 静态目录")

    def download(self, bucket: str, object_name: str) -> bytes:
        url = self._full_url(object_name)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.content

    def download_stream(
        self, bucket: str, object_name: str, chunk_size: int = _CHUNK_SIZE
    ) -> Iterator[bytes]:
        url = self._full_url(object_name)
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    yield chunk

    def delete(self, bucket: str, object_name: str) -> None:
        # nginx-static 后端不管理删除，文件由外部部署/脚本管理
        logger.warning("nginx-static 后端不支持删除操作，object_name=%s", object_name)

    def exists(self, bucket: str, object_name: str) -> bool:
        url = self._full_url(object_name)
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code == 200
        except Exception:
            return False

    def get_size(self, bucket: str, object_name: str) -> int | None:
        url = self._full_url(object_name)
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            content_length = response.headers.get("Content-Length")
            return int(content_length) if content_length else None
        except Exception:
            return None

    def ensure_bucket(self, bucket: str) -> None:
        # nginx-static 后端无 bucket 概念
        pass

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        # nginx-static 后端不支持列目录（nginx autoindex 默认关闭）
        raise NotImplementedError("nginx-static 后端不支持列出对象")
