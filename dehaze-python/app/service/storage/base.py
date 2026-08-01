"""
存储服务抽象基类

定义统一的存储操作接口，支持多种存储后端切换。
URL 永远运行时拼接（baseUrl + object_name），不落库。
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional


class StorageService(ABC):
    """
    存储服务抽象基类

    定义文件存储的标准操作接口。
    具体实现通过 StorageFactory 根据配置选择。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """存储后端标识（minio/local/nginx-static）"""

    @property
    @abstractmethod
    def base_url(self) -> str:
        """该后端的 baseUrl（完整 URL，运行时拼接 object_name 用）"""

    @abstractmethod
    def upload(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str,
    ) -> None:
        """上传文件（同步方法，需在线程池中调用）"""

    @abstractmethod
    def download(self, bucket: str, object_name: str) -> bytes:
        """下载文件（同步方法，需在线程池中调用）"""

    @abstractmethod
    def download_stream(self, bucket: str, object_name: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        """流式下载文件（同步生成器，需在线程池中调用）"""

    @abstractmethod
    def delete(self, bucket: str, object_name: str) -> None:
        """删除文件（同步方法，需在线程池中调用）"""

    @abstractmethod
    def exists(self, bucket: str, object_name: str) -> bool:
        """检查文件是否存在"""

    @abstractmethod
    def get_size(self, bucket: str, object_name: str) -> Optional[int]:
        """获取文件大小（字节），不存在返回 None"""

    def get_url(self, object_name: str) -> str:
        """生成文件访问 URL（运行时拼接，不落库）。

        url = baseUrl.rstrip("/") + "/" + object_name
        """
        return f"{self.base_url.rstrip('/')}/{object_name}"

    @abstractmethod
    def ensure_bucket(self, bucket: str) -> None:
        """确保存储桶存在（不存在则创建）"""

    @abstractmethod
    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """列出指定前缀下的所有对象名"""
