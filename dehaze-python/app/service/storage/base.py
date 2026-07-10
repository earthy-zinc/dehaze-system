"""
存储服务抽象基类

定义统一的存储操作接口，支持多种存储后端切换。
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional


class StorageService(ABC):
    """
    存储服务抽象基类

    定义文件存储的标准操作接口。
    具体实现通过 StorageFactory 根据配置选择。
    """

    @abstractmethod
    def upload(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str,
    ) -> None:
        """
        上传文件（同步方法，需在线程池中调用）

        Args:
            bucket: 存储桶名称
            object_name: 对象名称
            data: 文件内容
            content_type: MIME 类型
        """

    @abstractmethod
    def download(self, bucket: str, object_name: str) -> bytes:
        """
        下载文件（同步方法，需在线程池中调用）

        Args:
            bucket: 存储桶名称
            object_name: 对象名称

        Returns:
            文件内容
        """

    @abstractmethod
    def download_stream(self, bucket: str, object_name: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        """
        流式下载文件（同步生成器，需在线程池中调用）

        Args:
            bucket: 存储桶名称
            object_name: 对象名称
            chunk_size: 分块大小

        Yields:
            文件内容分块
        """

    @abstractmethod
    def delete(self, bucket: str, object_name: str) -> None:
        """
        删除文件（同步方法，需在线程池中调用）

        Args:
            bucket: 存储桶名称
            object_name: 对象名称
        """

    @abstractmethod
    def exists(self, bucket: str, object_name: str) -> bool:
        """
        检查文件是否存在

        Args:
            bucket: 存储桶名称
            object_name: 对象名称

        Returns:
            是否存在
        """

    @abstractmethod
    def get_size(self, bucket: str, object_name: str) -> Optional[int]:
        """
        获取文件大小

        Args:
            bucket: 存储桶名称
            object_name: 对象名称

        Returns:
            文件大小（字节），不存在返回 None
        """

    @abstractmethod
    def get_url(self, bucket: str, object_name: str) -> str:
        """
        生成文件访问 URL

        Args:
            bucket: 存储桶名称
            object_name: 对象名称

        Returns:
            文件访问 URL
        """

    @abstractmethod
    def ensure_bucket(self, bucket: str) -> None:
        """
        确保存储桶存在（不存在则创建）

        Args:
            bucket: 存储桶名称
        """

    @abstractmethod
    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """
        列出指定前缀下的所有对象名

        Args:
            bucket: 存储桶名称
            prefix: 前缀过滤

        Returns:
            对象名列表
        """
