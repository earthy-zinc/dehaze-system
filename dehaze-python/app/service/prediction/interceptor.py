"""
预测主流程拦截器（对齐 Java PredictionInterceptor / PredictionInterceptorChain）

命中（返回非 None）则短路主流程，不调用算法；
未命中（返回 None）则继续走主流程。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_file import SysFile

logger = logging.getLogger(__name__)


@dataclass
class PredictionContext:
    algorithm: SysAlgorithm
    file_id: Optional[int] = None
    image_url: Optional[str] = None
    origin_file: Optional[SysFile] = None
    image_md5: Optional[str] = None
    params: Optional[dict] = None
    start_time_ms: int = 0


@dataclass
class InterceptedResult:
    result_url: str
    result_md5: str
    result_file_id: Optional[int] = None


class PredictionInterceptor(ABC):
    """预测拦截器抽象基类"""

    @abstractmethod
    async def intercept(self, context: PredictionContext) -> Optional[InterceptedResult]:
        """返回非 None 表示命中，主流程短路；返回 None 表示继续"""
        ...


class PredictionInterceptorChain:
    """责任链：按注册顺序执行，第一个命中即短路"""

    def __init__(self, interceptors: list[PredictionInterceptor]):
        self._interceptors = interceptors

    async def intercept(self, context: PredictionContext) -> Optional[InterceptedResult]:
        for interceptor in self._interceptors:
            try:
                result = await interceptor.intercept(context)
                if result is not None:
                    logger.info(
                        "预测拦截器命中: %s -> resultUrl=%s",
                        interceptor.__class__.__name__, result.result_url,
                    )
                    return result
            except Exception as e:
                logger.warning(
                    "预测拦截器执行异常，跳过: %s - %s",
                    interceptor.__class__.__name__, e, exc_info=True,
                )
        return None
