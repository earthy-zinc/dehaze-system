"""
文件服务

提供文件上传、下载、删除等功能。
URL 不落库，永远运行时拼接（StorageService.get_url）。
"""

import asyncio
import hashlib
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime
from urllib.parse import quote

from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_file import SysFile
from app.repository.file_repository import file_repository
from app.service.storage.executor import storage_executor
from app.service.storage.factory import get_storage_by_name, get_storage_service
from app.utils.file import convert_size

logger = logging.getLogger(__name__)

# 文件名安全校验正则：禁止路径遍历、空字节、管道等特殊字符
_UNSAFE_FILENAME_PATTERN = re.compile(r'[\\/:*?"<>|\x00-\x1f]|\.\./')

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


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

    filename = filename.replace("\\", "/").split("/")[-1].strip()

    if _UNSAFE_FILENAME_PATTERN.search(filename):
        raise BusinessException(ResultCode.PARAM_ERROR, "文件名包含非法字符")

    if not filename:
        raise BusinessException(ResultCode.PARAM_ERROR, "文件名无效")

    return filename


class FileService:
    """文件服务类（异步版本）"""

    async def upload_file(self, 
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
        filename = sanitize_filename(filename)

        file_md5 = hashlib.md5(content).hexdigest()
        file_size = len(content)

        existing_file = await file_repository.get_by_md5(db, file_md5)
        if existing_file:
            return existing_file

        file_extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
        object_name = generate_object_name(file_md5, file_extension)

        # 上传到默认存储后端（在线程池中执行，避免阻塞事件循环）
        storage_service = get_storage_service()
        bucket_name = settings.MINIO_BUCKET

        def _sync_upload():
            """同步上传操作"""
            storage_service.upload(bucket_name, object_name, content, content_type)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(storage_executor, _sync_upload)
        except Exception as e:
            logger.error("文件上传到存储服务失败: %s", e, exc_info=True)
            raise BusinessException(
                ResultCode.FILE_STORAGE_ERROR, f"文件存储失败: {str(e)}"
            ) from None

        # 构造 SysFile 实体对象（URL 不落库，运行时拼接）
        new_file = SysFile(
            type=file_extension,
            name=filename,
            object_name=object_name,
            storage=storage_service.name,
            size=convert_size(file_size),
            size_bytes=file_size,
            md5=file_md5,
        )

        # 主动 upsert：冲突时复活 deleted=0，返回已有或新记录
        created_file = await file_repository.upsert_by_md5(
            db,
            md5=new_file.md5,
            type=file_extension,
            name=new_file.name,
            object_name=new_file.object_name,
            storage=new_file.storage,
            size=new_file.size,
            size_bytes=file_size,
        )

        return created_file

    async def delete_file_with_storage(self, db: AsyncSession, file_id: int) -> None:
        """
        删除文件记录及物理存储

        DB 记录删除后，物理文件删除为 best-effort（失败仅记录日志，
        由孤儿文件清理任务兜底）。

        Args:
            db: 异步数据库会话
            file_id: 文件 ID

        Raises:
            BusinessException: 文件不存在
        """
        file_info = await file_repository.get_by_id(db, file_id)

        if not file_info:
            raise BusinessException("不存在当前文件")

        object_name = file_info.object_name

        # 删除数据库记录（事务由 get_db() 在请求边界统一提交）
        await file_repository.soft_delete_by_ids(db, [file_id])

        # 联动失效：文件删除时，将直接/间接引用该文件的 AI 产物标记失效
        from app.service.ai_artifact_service import ai_artifact_service

        await ai_artifact_service.mark_invalid_for_file(db, file_id)

        # 从存储中删除文件（在线程池中异步执行，不阻塞事件循环）
        storage_service = get_storage_by_name(
            getattr(file_info, "storage", None) or settings.FILE_STORAGE_TYPE
        )
        bucket_name = settings.MINIO_BUCKET

        def _sync_remove():
            try:
                storage_service.delete(bucket_name, object_name)
            except Exception as e:
                # 存储删除失败仅记录日志，不影响数据库删除结果
                logger.warning("物理文件删除失败 [%s]: %s", object_name, e)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(storage_executor, _sync_remove)
        except Exception as e:
            logger.warning("物理文件删除异常 [%s]: %s", object_name, e)

    async def get_file_by_md5(self, db: AsyncSession, md5: str) -> SysFile | None:
        """
        根据 MD5 获取文件记录

        Args:
            db: 异步数据库会话
            md5: 文件 MD5 值

        Returns:
            SysFile 或 None
        """
        return await file_repository.get_by_md5(db, md5)

    async def get_file_by_id(self, db: AsyncSession, file_id: int) -> SysFile | None:
        """
        根据 ID 获取文件记录

        Args:
            db: 异步数据库会话
            file_id: 文件 ID

        Returns:
            SysFile 或 None
        """
        return await file_repository.get_by_id(db, file_id)

    async def get_file_by_object_name(self, db: AsyncSession, object_name: str) -> SysFile | None:
        """
        根据对象名称获取文件记录

        Args:
            db: 异步数据库会话
            object_name: MinIO 对象名称

        Returns:
            SysFile 或 None
        """
        return await file_repository.get_by_object_name(db, object_name)

    async def get_file_page(self, 
        db: AsyncSession,
        page: int,
        size: int,
        keywords: str | None = None,
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

    async def download_file_stream(self, 
        object_name: str, storage: str = "minio"
    ) -> AsyncIterator[bytes]:
        """
        从指定存储后端流式下载文件（避免大文件 OOM）

        Args:
            object_name: 对象名称
            storage: 存储后端标识（minio/local/nginx-static）

        Yields:
            文件内容分块

        Raises:
            BusinessException: 文件不存在或下载失败
        """
        storage_service = get_storage_by_name(storage)
        bucket_name = settings.MINIO_BUCKET

        # nginx-static 后端是 HTTP GET 取流，可直接通过 requests 流式迭代；
        # minio/local 也是同步读取，统一通过生产者-消费者队列桥接到异步生成器
        _SENTINEL = object()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        loop = asyncio.get_running_loop()

        def _producer():
            """在工作线程中分块读取对象并推入队列（带背压）"""
            try:
                for chunk in storage_service.download_stream(bucket_name, object_name):
                    fut = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                    fut.result()  # 等待队列有空间（背压）
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop).result()

        loop.run_in_executor(storage_executor, _producer)

        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    logger.error("文件下载失败 [%s]: %s", object_name, item)
                    raise BusinessException(ResultCode.FILE_NOT_FOUND, "文件下载失败")
                yield item
        except BusinessException:
            raise
        except Exception as e:
            logger.error("文件下载失败 [%s]: %s", object_name, e, exc_info=True)
            raise BusinessException(ResultCode.FILE_NOT_FOUND, "文件下载失败") from None

    def stream_file_response(self, object_name: str, storage: str = "minio") -> StreamingResponse:
        """
        从指定存储后端流式返回文件内容（带下载用 Content-Disposition 头）

        Args:
            object_name: 对象名称
            storage: 存储后端标识（minio/local/nginx-static）

        Returns:
            可直接作为 FastAPI 路由返回值的 StreamingResponse
        """
        filename = object_name.rsplit("/", 1)[-1] or "download"
        ascii_filename = filename.encode("ascii", "ignore").decode("ascii") or "download"
        encoded_filename = quote(filename)
        content_disposition = (
            f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}"
        )
        return StreamingResponse(
            self.download_file_stream(object_name, storage=storage),
            media_type="application/octet-stream",
            headers={"Content-Disposition": content_disposition},
        )

    async def get_file_stat(self, object_name: str) -> int | None:
        """
        获取存储中文件的大小（字节）

        Args:
            object_name: MinIO 对象名称

        Returns:
            文件大小（字节），获取失败返回 None
        """
        storage_service = get_storage_service()
        bucket_name = settings.MINIO_BUCKET

        def _sync_stat():
            return storage_service.get_size(bucket_name, object_name)

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(storage_executor, _sync_stat)
        except Exception:
            return None

    async def ensure_bucket_exists(self, ) -> None:
        """启动时确保存储 Bucket 存在（仅 MinIO 模式）"""
        if settings.FILE_STORAGE_TYPE != "minio":
            logger.info("非 MinIO 模式，跳过 Bucket 检查")
            return

        storage_service = get_storage_service()
        bucket_name = settings.MINIO_BUCKET

        def _sync_check():
            storage_service.ensure_bucket(bucket_name)
            logger.info("MinIO Bucket 已就绪: %s", bucket_name)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(storage_executor, _sync_check)
        except Exception as e:
            logger.warning("检查/创建 MinIO Bucket 失败: %s", e)


file_service = FileService()
