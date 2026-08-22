"""
预测主流程拦截器（对齐 Java PredictionInterceptor / PredictionInterceptorChain）

命中（返回非 None）则短路主流程，不调用算法；
未命中（返回 None）则继续走主流程。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_file import SysFile

logger = logging.getLogger(__name__)


@dataclass
class PredictionContext:
    algorithm: SysAlgorithm
    file_id: int | None = None
    image_url: str | None = None
    origin_file: SysFile | None = None
    image_md5: str | None = None
    params: dict | None = None
    start_time_ms: int = 0


@dataclass
class InterceptedResult:
    result_url: str
    result_md5: str
    result_file_id: int | None = None


class PredictionInterceptor(ABC):
    """预测拦截器抽象基类"""

    @abstractmethod
    async def intercept(self, context: PredictionContext) -> InterceptedResult | None:
        """返回非 None 表示命中，主流程短路；返回 None 表示继续"""
        ...


class PredictionInterceptorChain:
    """责任链：按注册顺序执行，第一个命中即短路"""

    def __init__(self, interceptors: list[PredictionInterceptor]):
        self._interceptors = interceptors

    async def intercept(self, context: PredictionContext) -> InterceptedResult | None:
        for interceptor in self._interceptors:
            try:
                result = await interceptor.intercept(context)
                if result is not None:
                    logger.debug(
                        "预测拦截器命中: %s -> resultUrl=%s",
                        interceptor.__class__.__name__,
                        result.result_url,
                    )
                    return result
            except Exception as e:
                logger.warning(
                    "预测拦截器执行异常，跳过: %s - %s",
                    interceptor.__class__.__name__,
                    e,
                    exc_info=True,
                )
        return None
