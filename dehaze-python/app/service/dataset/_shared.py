"""数据集域共享件：logger、跨类文件 VO 组装 helper 与图片内容校验"""

import io
import logging
from typing import Any

import PIL.Image

from app.core.code import ResultCode
from app.core.exceptions import BusinessException

logger = logging.getLogger(__name__)

# 数据项图片仅允许真实位图格式：非图片扩展名（如 html/svg）经文件 URL 以
# text/html 渲染会形成存储型 XSS，必须在入库前拦截
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}


def validate_image_content(filename: str, content: bytes) -> None:
    """校验数据项上传文件为真实图片：扩展名白名单 + PIL 可解析。

    file_service 的魔数校验只覆盖已知图片扩展名，非图片扩展名会直接放行，
    因此数据项图片上传必须在此再做一道白名单 + 解析校验。
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in IMAGE_EXTENSIONS:
        raise BusinessException(ResultCode.PARAM_ERROR, "仅支持图片格式（jpg/png/gif/bmp/webp）")

    try:
        with PIL.Image.open(io.BytesIO(content)) as img:
            img.verify()
    except Exception:
        raise BusinessException(ResultCode.PARAM_ERROR, "文件内容不是有效的图片") from None


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
