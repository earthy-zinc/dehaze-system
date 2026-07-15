import hashlib
import math
import os
from io import BytesIO


def convert_size(size_bytes) -> str:
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def calculate_bytes_md5(bytes_io: BytesIO) -> str:
    # 重置 BytesIO 对象的指针到开头
    bytes_io.seek(0)

    hash_md5 = hashlib.md5()
    while True:
        chunk = bytes_io.read(4096)
        if not chunk:
            break
        hash_md5.update(chunk)

    # 重置 BytesIO 对象的指针到开头
    bytes_io.seek(0)
    return hash_md5.hexdigest()


def get_file_size(filepath: str) -> str:
    """获取文件大小（调用方应已通过 os.path.isfile 校验）"""
    size = os.path.getsize(filepath)
    return convert_size(size)
