"""数据集域共享件：logger 与跨类文件 VO 组装 helper"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _build_file_vo(item_file, file_obj) -> dict[str, Any]:
    """构建图片文件 VO（统一字段命名，对齐 SDK ImageUrlVO）。
    url/thumbnailUrl 运行时拼接（baseUrl + object_name），不落库。
    用于 item_file_service 和 dataset_item_service 的所有文件响应。"""
    from app.service.storage.factory import get_storage_by_name

    file_format = None
    if file_obj and file_obj.name and "." in file_obj.name:
        file_format = file_obj.name.rsplit(".", 1)[-1].lower()
    elif file_obj and file_obj.type:
        file_format = file_obj.type.lower()

    file_url = None
    if file_obj and file_obj.object_name:
        storage_service = get_storage_by_name(file_obj.storage)
        file_url = storage_service.get_url(file_obj.object_name)

    return {
        "id": item_file.id,
        "itemId": item_file.item_id,
        "fileId": item_file.file_id,
        "type": item_file.type,
        "sceneType": item_file.scene_type,
        "hazeLevel": item_file.haze_level,
        "description": item_file.description,
        "url": file_url,
        "thumbnailUrl": file_url,
        "fileName": file_obj.name if file_obj else None,
        "name": file_obj.name if file_obj else None,
        "sizeBytes": file_obj.size_bytes if file_obj else None,
        "size": file_obj.size_bytes if file_obj else None,
        "formattedSize": file_obj.size if file_obj else None,
        "format": file_format,
        "md5": file_obj.md5 if file_obj else None,
    }

