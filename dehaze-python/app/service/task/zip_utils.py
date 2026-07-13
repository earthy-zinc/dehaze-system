"""
ZIP 导出公共工具

提供数据项到 ZIP 文件的打包和上传能力，供各策略复用。
"""

from __future__ import annotations

import logging
import os
import uuid
import zipfile
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.config import settings
from app.core.exceptions import TaskCancelledException
from app.models.entity.sys_dataset import SysItemFile
from app.models.entity.sys_task import SysTask
from app.repository.dataset_repository import dataset_repository
from app.repository.file_repository import file_repository
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], Awaitable[None]]
CancelChecker = Callable[[], Awaitable[bool]]


async def export_items_to_zip(
    db: AsyncSession,
    sys_task: SysTask,
    item_ids: List[int],
    zip_name: str,
    options: Dict[str, Any],
    progress_callback: ProgressCallback,
    cancel_checker: CancelChecker,
) -> Optional[str]:
    """
    将数据项导出为 ZIP 文件并上传到 MinIO

    Args:
        db: 数据库会话
        sys_task: 任务实体
        item_ids: 数据项 ID 列表
        zip_name: ZIP 文件基础名称
        options: 导出选项
        progress_callback: 进度回调
        cancel_checker: 取消检测

    Returns:
        MinIO 预签名下载 URL（或本地路径）
    """
    # 检查取消标志位
    if await cancel_checker():
        raise TaskCancelledException()

    # 创建临时目录
    temp_dir = os.path.join(settings.TEMP_DIR_RESOLVED, 'export')
    os.makedirs(temp_dir, exist_ok=True)

    zip_filename = f"export_{zip_name}_{uuid.uuid4().hex[:8]}.zip"
    zip_path = os.path.join(temp_dir, zip_filename)

    # MinIO 对象名称
    object_name = f"exports/{sys_task.task_id}/{zip_name}.zip"

    # 解析导出选项
    structure = options.get('structure', 'by_item')
    include_types = options.get('includeTypes')
    include_thumbnail = options.get('includeThumbnail', False)

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zos:
            total_files = 0
            processed_files = 0

            # 预计算文件总数
            for item_id in item_ids:
                item_files = await dataset_repository.get_item_files_by_item_id(db, item_id)
                total_files += len(item_files)
                if include_thumbnail:
                    total_files += len(item_files)

            sys_task.total_files = total_files
            await db.flush()

            # 遍历每个数据项
            for item_id in item_ids:
                # 检查取消标志位
                if await cancel_checker():
                    raise TaskCancelledException()

                item = await dataset_repository.get_item_by_id(db, item_id)
                if item is None:
                    logger.warning("数据项不存在: itemId=%s", item_id)
                    continue

                item_files = await dataset_repository.get_item_files_by_item_id(db, item_id)

                for item_file in item_files:
                    if _should_include_type(include_types, item_file.type):
                        await _add_file_to_zip(
                            db, zos, item_file, structure, item.name or f"item_{item.id}", None, False
                        )
                        processed_files += 1
                        await progress_callback(processed_files, total_files)

                    if include_thumbnail:
                        await _add_file_to_zip(
                            db, zos, item_file, structure, item.name or f"item_{item.id}", "thumbnail", True
                        )
                        processed_files += 1
                        await progress_callback(processed_files, total_files)

        logger.info(
            "ZIP文件创建成功: file=%s, totalFiles=%s", zip_path, processed_files)

        # 上传到 MinIO 并生成预签名 URL
        download_url = await _upload_to_minio(zip_path, object_name)

        logger.info(
            "文件上传完成: objectName=%s, downloadUrl=%s", object_name, download_url)
        return download_url

    except TaskCancelledException:
        raise
    except Exception as e:
        logger.error("导出文件失败", exc_info=True)
        from app.core.exceptions import BusinessException
        raise BusinessException(f"导出文件失败: {e}")
    finally:
        # 清理临时文件
        if os.path.exists(zip_path):
            os.remove(zip_path)


def _should_include_type(include_types: Optional[List[str]], file_type: str) -> bool:
    """判断是否应该包含该类型"""
    if not include_types:
        return True
    return file_type in include_types


async def _add_file_to_zip(
    db: AsyncSession,
    zos: zipfile.ZipFile,
    item_file: SysItemFile,
    structure: str,
    item_name: str,
    subfolder: Optional[str],
    is_thumbnail: bool,
) -> None:
    """
    添加文件到 ZIP

    从 MinIO 下载文件内容后写入 ZIP。
    """
    # 确定要下载的文件 ID
    file_id = item_file.thumbnail_file_id if is_thumbnail else item_file.file_id
    if file_id is None:
        return

    file_obj = await file_repository.get_by_id(db, file_id)
    if file_obj is None:
        logger.warning("文件不存在: fileId=%s", file_id)
        return

    # 构建 ZIP 条目路径
    ext = _get_extension(file_obj.name) or ".jpg"
    if structure == 'by_item':
        if subfolder:
            entry_path = f"{item_name}/{subfolder}/{item_file.id}{ext}"
        else:
            entry_path = f"{item_name}/{item_file.id}{ext}"
    else:
        if subfolder:
            entry_path = f"{subfolder}/{item_file.id}{ext}"
        else:
            entry_path = f"{item_file.id}{ext}"

    # 从 MinIO 下载文件字节并写入 ZIP
    file_bytes = await _download_from_minio(file_obj.object_name)
    if file_bytes:
        zos.writestr(entry_path, file_bytes)
    else:
        logger.warning("从存储下载文件失败，跳过: objectName=%s", file_obj.object_name)


def _get_extension(filename: str) -> str:
    """从文件名中提取扩展名"""
    if '.' in filename:
        return '.' + filename.rsplit('.', 1)[-1]
    return ''


async def _download_from_minio(object_name: str) -> Optional[bytes]:
    """
    从 MinIO 下载文件字节（在线程池中执行，避免阻塞事件循环）

    Returns:
        文件字节数据，失败返回 None
    """
    import asyncio

    from app.service.file_service import _minio_executor, get_minio_client
    from app.config import settings

    client = get_minio_client()
    bucket_name = settings.MINIO_BUCKET_NAME

    def _sync_download() -> Optional[bytes]:
        response = None
        try:
            response = client.get_object(bucket_name, object_name)
            return response.read()
        except Exception as e:
            logger.error("MinIO 下载失败: objectName=%s, error=%s", object_name, e)
            return None
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_minio_executor, _sync_download)
    except Exception as e:
        logger.error("MinIO 下载执行失败: objectName=%s, error=%s", object_name, e)
        return None


async def _upload_to_minio(local_path: str, object_name: str) -> str:
    """
    上传文件到 MinIO 并生成预签名 URL（在线程池中执行，避免阻塞事件循环）

    Args:
        local_path: 本地文件路径
        object_name: MinIO 对象名称

    Returns:
        预签名下载 URL
    """
    import asyncio
    from datetime import timedelta

    from app.service.file_service import _minio_executor, get_minio_client
    from app.config import settings

    client = get_minio_client()
    bucket_name = settings.MINIO_BUCKET_NAME

    def _sync_upload() -> str:
        client.fput_object(
            bucket_name,
            object_name,
            local_path,
            content_type="application/zip",
        )
        # 生成 24 小时有效的预签名 URL
        return client.presigned_get_object(
            bucket_name,
            object_name,
            expires=timedelta(hours=24),
        )

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_minio_executor, _sync_upload)
    except Exception as e:
        logger.warning("MinIO 上传失败，降级为本地路径: %s", e)
        # 降级：返回本地下载路径（需要 Web 服务器提供 static 路由）
        return f"/downloads/{object_name}"
