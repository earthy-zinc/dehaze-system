"""
WPXNet 系列模型预查询拦截器（对齐 Java WpxNetPredictionInterceptor）

当模型为 WPXNet 系（model_id 前缀 `wpxnet-`，由阶段 1 迁移自算法目录名规范 slug）
且原始图 MD5 在 sys_wpx_file 表中存在映射时，直接返回已处理好的去雾图，跳过推理。

前置条件：sys_wpx_file 表已由 scripts/init_wpx_file.py 写入映射数据。
"""

import logging

from app.database import async_session_factory
from app.repository.file_repository import file_repository
from app.repository.wpx_file_repository import wpx_file_repository
from app.service.prediction.interceptor import (
    InterceptedResult,
    PredictionContext,
    PredictionInterceptor,
)
from app.service.storage.factory import get_storage_by_name

logger = logging.getLogger(__name__)

WPXNET_MODEL_PREFIX = "wpxnet-"


class WpxNetPredictionInterceptor(PredictionInterceptor):
    """WPXNet 预查询拦截器"""

    async def intercept(self, context: PredictionContext) -> InterceptedResult | None:
        if not context.model.model_id.startswith(WPXNET_MODEL_PREFIX):
            return None

        async with async_session_factory() as db:
            origin_file = context.origin_file
            if origin_file is None or not origin_file.md5:
                return None
            origin_md5 = origin_file.md5

            wpx_file = await wpx_file_repository.get_by_origin_md5(db, origin_md5)
            if wpx_file is None or wpx_file.new_file_id is None:
                logger.debug(
                    "WPXNet 命中模型但未找到映射: model=%s, originMd5=%s",
                    context.model.model_id,
                    origin_md5,
                )
                return None

            new_file = await file_repository.get_by_id(db, wpx_file.new_file_id)
            if new_file is None:
                logger.warning(
                    "WPXNet 映射的 new_file_id 不存在: wpxFileId=%s, newFileId=%s",
                    wpx_file.id,
                    wpx_file.new_file_id,
                )
                return None

            logger.debug(
                "WPXNet 预查询命中: model=%s, originMd5=%s, resultFileId=%s",
                context.model.model_id,
                origin_md5,
                new_file.id,
            )
            # 返回 result_url：通过 storage.get_url 运行时拼接（不落库）
            storage_service = get_storage_by_name(new_file.storage)
            return InterceptedResult(
                result_url=storage_service.get_url(new_file.object_name),
                result_md5=new_file.md5,
                result_file_id=new_file.id,
            )
