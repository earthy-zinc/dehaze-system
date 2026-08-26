"""预测结果上传存储：上传到默认存储后端（与文件管理模块共享存储桶）。"""

import asyncio
import io
import logging
import time
from datetime import datetime

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.storage.executor import storage_executor
from app.service.storage.factory import get_storage_service

logger = logging.getLogger(__name__)


async def upload_result(result_bytes: io.BytesIO, algorithm_name: str) -> str:
    """上传预测结果到存储。

    返回 object_name（对象键），URL 由响应层通过 storage.get_url 拼接。
    """
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{algorithm_name}_{int(time.time() * 1000)}.png"
    object_name = f"predictions/{date_str}/{filename}"

    data = result_bytes.getvalue()
    storage_service = get_storage_service()
    bucket_name = settings.MINIO_BUCKET
    loop = asyncio.get_running_loop()

    def _sync_upload():
        storage_service.upload(bucket_name, object_name, data, "image/png")

    try:
        await loop.run_in_executor(storage_executor, _sync_upload)
    except Exception as e:
        logger.error("预测结果上传存储失败: %s", e, exc_info=True)
        raise BusinessException(
            ResultCode.FILE_STORAGE_ERROR, f"结果存储失败: {str(e)}"
        ) from None

    logger.debug("预测结果已上传到存储: %s", object_name)
    return object_name
