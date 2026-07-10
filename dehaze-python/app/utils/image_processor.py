"""
图片处理工具

提供图片格式校验、尺寸信息解析等功能。
"""

import logging
import struct
from typing import Optional

logger = logging.getLogger(__name__)

# 图片文件头 Magic Bytes 映射
# 参考: https://en.wikipedia.org/wiki/List_of_file_signatures
_IMAGE_SIGNATURES = {
    b'\xFF\xD8\xFF': 'jpg',      # JPEG
    b'\x89PNG\r\n\x1a\n': 'png',  # PNG
    b'GIF87a': 'gif',             # GIF87a
    b'GIF89a': 'gif',             # GIF89a
    b'RIFF': 'webp',             # WebP (RIFF....WEBP)
    b'BM': 'bmp',                # BMP
}

# 支持的图片扩展名集合
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp", "bmp", "svg"}


def detect_image_type(data: bytes) -> Optional[str]:
    """
    通过 Magic Bytes 检测图片类型

    Args:
        data: 文件内容（至少前 16 字节）

    Returns:
        图片类型（如 'jpg', 'png'），非图片返回 None
    """
    if len(data) < 4:
        return None

    for signature, img_type in _IMAGE_SIGNATURES.items():
        if data[:len(signature)] == signature:
            # WebP 需额外检查 RIFF 后的 WEBP 标识
            if img_type == 'webp':
                if len(data) >= 12 and data[8:12] == b'WEBP':
                    return 'webp'
                continue
            return img_type

    # SVG 检测（文本文件，检查 XML/SVG 标签）
    try:
        text_start = data[:256].decode('utf-8', errors='ignore').strip().lower()
        if '<svg' in text_start or '<?xml' in text_start:
            return 'svg'
    except Exception:
        pass

    return None


def is_image_file(data: bytes, declared_extension: str) -> bool:
    """
    校验文件是否为声明的图片类型（扩展名 + Magic Bytes 双重校验）

    Args:
        data: 文件内容
        declared_extension: 声明的文件扩展名

    Returns:
        True 如果文件确实是声明的图片类型
    """
    if declared_extension.lower() not in IMAGE_EXTENSIONS:
        return False

    detected = detect_image_type(data)
    if detected is None:
        return False

    # jpg/jpeg 视为同一类型
    declared_normalized = 'jpg' if declared_extension.lower() == 'jpeg' else declared_extension.lower()
    detected_normalized = 'jpg' if detected == 'jpeg' else detected

    return declared_normalized == detected_normalized


def get_image_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    """
    解析图片的宽高尺寸

    支持 JPEG, PNG, GIF, BMP 格式。

    Args:
        data: 图片文件内容

    Returns:
        (width, height) 元组，无法解析返回 None
    """
    if len(data) < 16:
        return None

    img_type = detect_image_type(data)

    try:
        if img_type == 'png':
            return _parse_png_dimensions(data)
        elif img_type == 'jpg':
            return _parse_jpeg_dimensions(data)
        elif img_type == 'gif':
            return _parse_gif_dimensions(data)
        elif img_type == 'bmp':
            return _parse_bmp_dimensions(data)
    except Exception as e:
        logger.debug(f"解析图片尺寸失败: {e}")

    return None


def _parse_png_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    """解析 PNG 宽高 (IHDR chunk)"""
    if len(data) < 24:
        return None
    width = struct.unpack('>I', data[16:20])[0]
    height = struct.unpack('>I', data[20:24])[0]
    return width, height


def _parse_jpeg_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    """解析 JPEG 宽高 (SOF0/SOF2 marker)"""
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        # SOF0 (0xC0) or SOF2 (0xC2) marker
        if marker in (0xC0, 0xC2):
            if i + 9 >= len(data):
                return None
            height = struct.unpack('>H', data[i + 5:i + 7])[0]
            width = struct.unpack('>H', data[i + 7:i + 9])[0]
            return width, height
        # Skip marker
        if i + 3 >= len(data):
            return None
        length = struct.unpack('>H', data[i + 2:i + 4])[0]
        i += 2 + length
    return None


def _parse_gif_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    """解析 GIF 宽高 (Logical Screen Descriptor)"""
    if len(data) < 10:
        return None
    width = struct.unpack('<H', data[6:8])[0]
    height = struct.unpack('<H', data[8:10])[0]
    return width, height


def _parse_bmp_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    """解析 BMP 宽高 (DIB Header)"""
    if len(data) < 26:
        return None
    width = struct.unpack('<i', data[18:22])[0]
    height = abs(struct.unpack('<i', data[22:26])[0])
    return width, height
