"""HTTP 边界载荷解析：JSON 表单字段与上传图像的合法性校验。"""

import json
from typing import Any

import PIL.Image
from fastapi import UploadFile

from app.errors import ModalError


def parse_json(raw: str, field: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ModalError("PARAM_ERROR", f"{field} 不是合法 JSON: {e}") from None


def open_image(upload: UploadFile) -> PIL.Image.Image:
    """解析上传图像（仅读文件头）；内容非法 → 422。"""
    try:
        return PIL.Image.open(upload.file)
    except (PIL.UnidentifiedImageError, OSError, ValueError) as e:
        raise ModalError("INVALID_IMAGE", f"图像内容非法: {str(e)[:200]}") from None


def pixel_count(image: PIL.Image.Image) -> int:
    width, height = image.size
    return width * height
