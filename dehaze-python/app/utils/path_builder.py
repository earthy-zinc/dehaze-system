"""
文件路径构建器

统一文件路径生成逻辑，支持多种组织策略。
"""

from datetime import datetime


def build_upload_path(md5: str, extension: str, prefix: str = "upload") -> str:
    """
    构建上传文件的存储路径

    格式: {prefix}/{YYYYMMDD}/{md5}.{ext}

    Args:
        md5: 文件 MD5 值
        extension: 文件扩展名（不含点号）
        prefix: 路径前缀

    Returns:
        对象存储路径
    """
    date_part = datetime.now().strftime("%Y%m%d")
    return f"{prefix}/{date_part}/{md5}.{extension}"


def build_temp_path(task_id: str, filename: str, prefix: str = "temp") -> str:
    """
    构建临时文件的存储路径

    格式: {prefix}/{task_id}/{filename}

    Args:
        task_id: 任务 ID
        filename: 文件名
        prefix: 路径前缀

    Returns:
        临时文件路径
    """
    return f"{prefix}/{task_id}/{filename}"


def build_result_path(task_id: str, filename: str, prefix: str = "result") -> str:
    """
    构建任务结果文件的存储路径

    格式: {prefix}/{YYYYMMDD}/{task_id}/{filename}

    Args:
        task_id: 任务 ID
        filename: 文件名
        prefix: 路径前缀

    Returns:
        结果文件路径
    """
    date_part = datetime.now().strftime("%Y%m%d")
    return f"{prefix}/{date_part}/{task_id}/{filename}"


def extract_extension(filename: str) -> str:
    """
    从文件名中提取扩展名

    Args:
        filename: 文件名

    Returns:
        小写扩展名，无扩展名返回 "bin"
    """
    if "." in filename:
        return filename.rsplit(".", 1)[-1].lower()
    return "bin"
