"""
数据集导出处理器

整合旧 DatasetExportStrategy / ItemDownloadStrategy / BatchDownloadStrategy / CustomExportStrategy
的 ZIP 打包逻辑。通过 queryParams 中的字段区分不同导出场景：
- datasetId: 导出整个数据集的所有数据项
- itemId: 导出单个数据项的文件
- itemIds: 批量导出多个数据项的文件
- filters: 按筛选条件导出数据集内数据项
"""

from __future__ import annotations

import io
import logging
import zipfile

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_dataset import SysDatasetItem, SysItemFile
from app.repository.dataset_repository import dataset_repository
from app.repository.file_repository import file_repository
from app.infrastructure.storage.minio_client import get_minio_client, minio_executor
from app.service.import_export.models import ExportContext, ExportFieldConfig
from app.service.import_export.registry import ExportHandler

logger = logging.getLogger(__name__)

STRUCTURE_BY_ITEM = "by_item"
THUMBNAIL_SUBFOLDER = "thumbnail"
DEFAULT_FILE_EXTENSION = ".jpg"
ZIP_BUFFER_SIZE = 8192


class DatasetExportHandler(ExportHandler):
    def get_module(self) -> str:
        return "dataset"

    def use_direct_export(self) -> bool:
        return True

    async def estimate_count(self, db: AsyncSession, query_params: dict) -> int:
        if not query_params:
            return 0
        items = await _resolve_items(db, query_params)
        if not items:
            return 0
        item_ids = [int(i.id) for i in items]
        item_files_map = await dataset_repository.get_item_files_by_item_ids(db, item_ids)
        return sum(len(item_files_map.get(iid, [])) for iid in item_ids)

    async def export(
        self,
        db: AsyncSession,
        ctx: ExportContext,
        output: io.BytesIO,
        progress_cb,
        cancel_cb,
    ) -> None:
        params = ctx.query_params or {}

        options = _ExportOptions.from_params(params)
        items = await _resolve_items(db, params)
        if not items:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "未找到可导出的数据项")

        item_ids = [int(i.id) for i in items]
        item_files_map = await dataset_repository.get_item_files_by_item_ids(db, item_ids)

        all_file_ids: set[int] = set()
        for iid in item_ids:
            for item_file in item_files_map.get(iid, []):
                if item_file.file_id is not None:
                    all_file_ids.add(item_file.file_id)
                if options.include_thumbnail and item_file.thumbnail_file_id is not None:
                    all_file_ids.add(item_file.thumbnail_file_id)
        file_objs = await file_repository.get_by_ids(db, list(all_file_ids))
        file_map = {int(f.id): f for f in file_objs}

        total_files = 0
        for iid in item_ids:
            count = len(item_files_map.get(iid, []))
            total_files += count
            if options.include_thumbnail:
                total_files += count

        await progress_cb(0, total_files)

        processed_files = 0
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zos:
            for item in items:
                if await cancel_cb():
                    break

                item_files = item_files_map.get(int(item.id), [])
                item_name = item.name or f"item_{item.id}"
                for item_file in item_files:
                    if await cancel_cb():
                        break

                    if _should_include_type(options.include_types, item_file.type):
                        await _add_file_to_zip(
                            zos, item_file, file_map, options.structure, item_name, None, False
                        )
                        processed_files += 1
                        await progress_cb(processed_files, total_files)

                    if options.include_thumbnail:
                        await _add_file_to_zip(
                            zos,
                            item_file,
                            file_map,
                            options.structure,
                            item_name,
                            THUMBNAIL_SUBFOLDER,
                            True,
                        )
                        processed_files += 1
                        await progress_cb(processed_files, total_files)

        logger.debug(
            "数据集导出完成: taskId=%s, itemCount=%s, fileCount=%s",
            ctx.task_id,
            len(items),
            processed_files,
        )

    def get_field_configs(self) -> list[ExportFieldConfig]:
        return [
            ExportFieldConfig(field="dataset_name", label="数据集名称", order=1, hidden=True),
            ExportFieldConfig(field="item_name", label="数据项名称", order=2, hidden=True),
            ExportFieldConfig(field="file_type", label="文件类型", order=3, hidden=True),
            ExportFieldConfig(field="file_name", label="文件名", order=4, hidden=True),
            ExportFieldConfig(field="file_size", label="文件大小", order=5, hidden=True),
        ]


async def _resolve_items(db: AsyncSession, params: dict) -> list[SysDatasetItem]:
    item_ids_obj = params.get("itemIds")
    if isinstance(item_ids_obj, list) and item_ids_obj:
        item_ids = [int(v) for v in item_ids_obj]
        return await dataset_repository.get_items_by_ids(db, item_ids)

    item_id_obj = params.get("itemId")
    if item_id_obj is not None:
        item_id = int(item_id_obj)
        item = await dataset_repository.get_item_by_id(db, item_id)
        return [item] if item else []

    dataset_id_obj = params.get("datasetId") or params.get("targetId")
    if dataset_id_obj is not None:
        dataset_id = int(dataset_id_obj)
        items = await dataset_repository.get_items_by_dataset_id(db, dataset_id)

        filters_obj = params.get("filters")
        if isinstance(filters_obj, dict):
            name = filters_obj.get("name")
            if isinstance(name, str) and name.strip():
                name_str = name.strip()
                items = [i for i in items if i.name and name_str in i.name]
        return items

    return []


def _should_include_type(include_types: list[str] | None, file_type: str | None) -> bool:
    if not include_types:
        return True
    return file_type in include_types


async def _add_file_to_zip(
    zos: zipfile.ZipFile,
    item_file: SysItemFile,
    file_map: dict,
    structure: str,
    item_name: str,
    subfolder: str | None,
    is_thumbnail: bool,
) -> None:
    file_id = item_file.thumbnail_file_id if is_thumbnail else item_file.file_id
    if file_id is None:
        return
    file_obj = file_map.get(file_id)
    if file_obj is None or not file_obj.object_name:
        logger.warning("文件不存在或 objectName 为空: fileId=%s", file_id)
        return

    entry_path = _build_zip_entry_path(structure, item_name, subfolder, item_file.id, file_obj.name)
    content = await _download_from_minio(file_obj.object_name)
    if content is None:
        logger.warning("从存储下载文件失败，跳过: objectName=%s", file_obj.object_name)
        return
    zos.writestr(entry_path, content)


def _build_zip_entry_path(
    structure: str,
    item_name: str,
    subfolder: str | None,
    file_id: int | None,
    file_name: str | None,
) -> str:
    extension = _get_extension(file_name or "")
    base_name = f"{file_id}{extension}"
    if structure == STRUCTURE_BY_ITEM:
        if subfolder:
            return f"{item_name}/{subfolder}/{base_name}"
        return f"{item_name}/{base_name}"
    if subfolder:
        return f"{subfolder}/{base_name}"
    return base_name


def _get_extension(filename: str) -> str:
    if "." not in filename:
        return DEFAULT_FILE_EXTENSION
    return "." + filename.rsplit(".", 1)[-1].lower()


async def _download_from_minio(object_name: str) -> bytes | None:
    import asyncio

    client = get_minio_client()
    bucket = settings.MINIO_BUCKET_NAME

    def _sync() -> bytes | None:
        response = None
        try:
            response = client.get_object(bucket, object_name)
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
        return await loop.run_in_executor(minio_executor, _sync)
    except Exception as e:
        logger.error("MinIO 下载执行失败: objectName=%s, error=%s", object_name, e)
        return None


class _ExportOptions:
    def __init__(
        self, structure: str, include_types: list[str] | None, include_thumbnail: bool
    ) -> None:
        self.structure = structure
        self.include_types = include_types
        self.include_thumbnail = include_thumbnail

    @staticmethod
    def from_params(params: dict) -> _ExportOptions:
        structure = STRUCTURE_BY_ITEM
        include_types: list[str] | None = None
        include_thumbnail = False

        options_obj = params.get("options")
        if isinstance(options_obj, dict):
            s = options_obj.get("structure")
            if isinstance(s, str):
                structure = s
            it = options_obj.get("includeTypes")
            if isinstance(it, list):
                include_types = [str(x) for x in it if isinstance(x, str)]
            it_flag = options_obj.get("includeThumbnail")
            if isinstance(it_flag, bool):
                include_thumbnail = it_flag

        return _ExportOptions(structure, include_types, include_thumbnail)
