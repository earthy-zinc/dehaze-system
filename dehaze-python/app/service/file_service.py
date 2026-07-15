"""
文件服务

提供文件上传、下载、删除等功能
"""

import asyncio
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from typing import AsyncIterator, Optional

from minio import Minio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_file import SysFile
from app.repository.file_repository import file_repository
from app.service.file_events import FileCreatedEvent, FileDeletedEvent, file_event_bus
from app.utils.file import convert_size

logger = logging.getLogger(__name__)

# MinIO 操作线程池（MinIO SDK 是同步的，需要在线程池中执行）
_minio_executor = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="minio-ops")

# MinIO 客户端单例
_minio_client: Optional[Minio] = None

# 文件名安全校验正则：禁止路径遍历、空字节、管道等特殊字符
_UNSAFE_FILENAME_PATTERN = re.compile(r'[\\/:*?"<>|\x00-\x1f]|\.\./')

# 文件下载分块大小 (1MB)
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


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


def generate_object_name(md5: str, extension: str) -> str:
    """生成 MinIO 对象名称"""
    return f"upload/{datetime.now().strftime('%Y%m%d')}/{md5}.{extension}"


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除危险字符

    Args:
        filename: 原始文件名

    Returns:
        安全的文件名

    Raises:
        BusinessException: 文件名无效
    """
    if not filename or not filename.strip():
        raise BusinessException(ResultCode.PARAM_ERROR, "文件名不能为空")

    # 移除路径部分，只保留文件名
    filename = filename.replace("\\", "/").split("/")[-1].strip()

    # 检查危险字符
    if _UNSAFE_FILENAME_PATTERN.search(filename):
        raise BusinessException(ResultCode.PARAM_ERROR, "文件名包含非法字符")

    if not filename:
        raise BusinessException(ResultCode.PARAM_ERROR, "文件名无效")

    return filename


def generate_file_url(object_name: str) -> str:
    """
    生成文件访问 URL

    优先使用 FILE_BASE_URL，其次使用 BASE_URL 域名，最后使用 MinIO 地址

    Args:
        object_name: 对象名称

    Returns:
        文件访问 URL
    """
    # 优先使用 FILE_BASE_URL 配置
    if settings.FILE_BASE_URL:
        base = settings.FILE_BASE_URL.rstrip("/")
        return f"{base}/{object_name}"

    # 兜底使用 MinIO 直连地址
    protocol = "https" if settings.MINIO_SECURE else "http"
    return f"{protocol}://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET_NAME}/{object_name}"


class FileService:
    """文件服务类（异步版本）"""

    @staticmethod
    async def upload_file(
        db: AsyncSession,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> SysFile:
        """
        上传文件到存储服务

        Args:
            db: 异步数据库会话
            filename: 原始文件名
            content: 文件内容（字节）
            content_type: 文件 MIME 类型

        Returns:
            SysFile: 文件记录

        Raises:
            BusinessException: 上传失败
        """
        # 清理文件名
        filename = sanitize_filename(filename)

        # 计算 MD5
        file_md5 = hashlib.md5(content).hexdigest()
        file_size = len(content)

        # 检查文件是否已存在（根据 MD5 去重）
        existing_file = await file_repository.get_by_md5(db, file_md5)
        if existing_file:
            return existing_file

        # 获取文件扩展名
        file_extension = filename.rsplit(
            ".", 1)[-1].lower() if "." in filename else "bin"
        object_name = generate_object_name(file_md5, file_extension)

        # 上传到 MinIO（在线程池中执行，避免阻塞事件循环）
        minio_client = get_minio_client()
        bucket_name = settings.MINIO_BUCKET_NAME

        def _sync_upload():
            """同步上传操作"""
            # 确保存储桶存在
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)

            # 上传文件
            minio_client.put_object(
                bucket_name,
                object_name,
                data=BytesIO(content),
                length=file_size,
                content_type=content_type,
            )

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_minio_executor, _sync_upload)
        except Exception as e:
            logger.error("文件上传到存储服务失败: %s", e, exc_info=True)
            raise BusinessException(
                ResultCode.FILE_STORAGE_ERROR, f"文件存储失败: {str(e)}")

        # 生成文件访问 URL
        file_url = generate_file_url(object_name)

        # 构造 SysFile 实体对象
        new_file = SysFile(
            type=file_extension,
            url=file_url,
            name=filename,
            object_name=object_name,
            size=convert_size(file_size),
            size_bytes=file_size,
            path=object_name,
            md5=file_md5,
        )

        # 创建数据库记录，处理并发 MD5 冲突
        # 使用 SAVEPOINT 避免回滚整个外层事务（upload_file 常被 dataset_service 在事务内调用）
        try:
            async with db.begin_nested():
                new_file = await file_repository.create(db, new_file)
        except IntegrityError:
            # 并发上传相同 MD5 文件导致唯一索引冲突，savepoint 自动回滚，外层事务不受影响
            existing_file = await file_repository.get_by_md5(db, file_md5)
            if existing_file:
                return existing_file
            # 如果重查仍未找到（理论上不应发生），抛出异常
            raise BusinessException(ResultCode.FILE_STORAGE_ERROR, "文件记录创建失败")

        # 发布文件创建事件
        file_event_bus.publish(FileCreatedEvent(
            file_id=new_file.id,
            filename=new_file.name,
            object_name=new_file.object_name,
            md5=new_file.md5,
            size_bytes=file_size,
        ))

        return new_file

    @staticmethod
    async def delete_file_with_storage(db: AsyncSession, file_id: int) -> None:
        """
        删除文件记录及物理存储

        DB 记录删除后，物理文件删除为 best-effort（失败仅记录日志，
        由孤儿文件清理任务 FILE_ORPHAN_CLEANUP_HOURS 兜底）。

        Args:
            db: 异步数据库会话
            file_id: 文件 ID

        Raises:
            BusinessException: 文件不存在
        """
        file_info = await file_repository.get_by_id(db, file_id)

        if not file_info:
            raise BusinessException("不存在当前文件")

        # 保存信息用于事件发布和存储删除
        object_name = file_info.object_name
        filename = file_info.name
        md5 = file_info.md5

        # 删除数据库记录（事务由 get_db() 在请求边界统一提交）
        await file_repository.delete_by_ids(db, [file_id])

        # 从存储中删除文件（在线程池中异步执行，不阻塞事件循环）
        minio_client = get_minio_client()
        bucket_name = settings.MINIO_BUCKET_NAME

        def _sync_remove():
            try:
                minio_client.remove_object(bucket_name, object_name)
            except Exception as e:
                # 存储删除失败仅记录日志，不影响数据库删除结果
                logger.warning("物理文件删除失败 [%s]: %s", object_name, e)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_minio_executor, _sync_remove)
        except Exception as e:
            logger.warning("物理文件删除异常 [%s]: %s", object_name, e)

        # 发布文件删除事件
        file_event_bus.publish(FileDeletedEvent(
            file_id=file_id,
            filename=filename,
            object_name=object_name,
            md5=md5,
        ))

    @staticmethod
    async def check_file_exists(db: AsyncSession, md5: str) -> bool:
        """
        检查文件是否已存在（根据 MD5）

        Args:
            db: 异步数据库会话
            md5: 文件 MD5 值

        Returns:
            bool: 文件是否存在
        """
        file_info = await file_repository.get_by_md5(db, md5)
        return file_info is not None

    @staticmethod
    async def get_file_by_md5(db: AsyncSession, md5: str) -> Optional[SysFile]:
        """
        根据 MD5 获取文件记录

        Args:
            db: 异步数据库会话
            md5: 文件 MD5 值

        Returns:
            SysFile 或 None
        """
        return await file_repository.get_by_md5(db, md5)

    @staticmethod
    async def get_file_by_id(db: AsyncSession, file_id: int) -> Optional[SysFile]:
        """
        根据 ID 获取文件记录

        Args:
            db: 异步数据库会话
            file_id: 文件 ID

        Returns:
            SysFile 或 None
        """
        return await file_repository.get_by_id(db, file_id)

    @staticmethod
    async def get_file_by_object_name(db: AsyncSession, object_name: str) -> Optional[SysFile]:
        """
        根据对象名称获取文件记录

        Args:
            db: 异步数据库会话
            object_name: MinIO 对象名称

        Returns:
            SysFile 或 None
        """
        return await file_repository.get_by_object_name(db, object_name)

    @staticmethod
    async def get_file_page(
        db: AsyncSession,
        page: int,
        size: int,
        keywords: Optional[str] = None,
    ) -> tuple[list[SysFile], int]:
        """
        分页查询文件列表

        Args:
            db: 异步数据库会话
            page: 页码（从 1 开始）
            size: 每页数量
            keywords: 搜索关键词（模糊匹配文件名）

        Returns:
            (items, total) 元组
        """
        return await file_repository.get_page(db, page, size, keywords)

    @staticmethod
    async def download_file_stream(object_name: str) -> AsyncIterator[bytes]:
        """
        从 MinIO 流式下载文件（避免大文件 OOM）

        Args:
            object_name: MinIO 对象名称

        Yields:
            文件内容分块

        Raises:
            BusinessException: 文件不存在或下载失败
        """
        minio_client = get_minio_client()
        bucket_name = settings.MINIO_BUCKET_NAME

        # 使用生产者-消费者队列实现真正的流式下载（避免大文件 OOM）
        _SENTINEL = object()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        loop = asyncio.get_running_loop()

        def _producer():
            """在工作线程中分块读取 MinIO 对象并推入队列（带背压）"""
            response = None
            try:
                response = minio_client.get_object(bucket_name, object_name)
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    fut = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                    fut.result()  # 等待队列有空间（背压）
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                if response:
                    response.close()
                    response.release_conn()
                asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop).result()

        # 启动生产者线程（非阻塞，在后台运行）
        loop.run_in_executor(_minio_executor, _producer)

        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    logger.error("文件下载失败 [%s]: %s", object_name, item)
                    raise BusinessException(
                        ResultCode.FILE_NOT_FOUND, "文件下载失败")
                yield item
        except BusinessException:
            raise
        except Exception as e:
            logger.error("文件下载失败 [%s]: %s", object_name, e, exc_info=True)
            raise BusinessException(
                ResultCode.FILE_NOT_FOUND, "文件下载失败")

    @staticmethod
    async def get_file_stat(object_name: str) -> Optional[int]:
        """
        获取存储中文件的大小（字节）

        Args:
            object_name: MinIO 对象名称

        Returns:
            文件大小（字节），获取失败返回 None
        """
        minio_client = get_minio_client()
        bucket_name = settings.MINIO_BUCKET_NAME

        def _sync_stat():
            stat = minio_client.stat_object(bucket_name, object_name)
            return stat.size

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(_minio_executor, _sync_stat)
        except Exception:
            return None

    @staticmethod
    async def ensure_bucket_exists() -> None:
        """启动时确保 MinIO Bucket 存在（仅 MinIO 模式）"""
        if settings.FILE_STORAGE_TYPE != "minio":
            logger.info("非 MinIO 模式，跳过 Bucket 检查")
            return

        minio_client = get_minio_client()
        bucket_name = settings.MINIO_BUCKET_NAME

        def _sync_check():
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
                logger.info("已自动创建 MinIO Bucket: %s", bucket_name)
            else:
                logger.info("MinIO Bucket 已存在: %s", bucket_name)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_minio_executor, _sync_check)
        except Exception as e:
            logger.warning("检查/创建 MinIO Bucket 失败: %s", e)
