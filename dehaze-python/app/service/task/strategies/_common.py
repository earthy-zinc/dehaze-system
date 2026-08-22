"""通用导入导出策略的公共工具"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from app.core.code import ResultCode
from app.core.exceptions import BusinessException, TaskCancelledException


def resolve_module(params: dict, task_type: str) -> str:
    """从参数或任务类型推导模块名（如 user_export -> user）"""
    module = params.get("module") or task_type.rsplit("_", 1)[0]
    if not module:
        raise BusinessException(ResultCode.TASK_PARAM_ERROR, "缺少模块参数 module")
    return module


def make_cancel_cb(
    cancel_checker: Callable[[], Awaitable[bool]],
) -> Callable[[], Awaitable[bool]]:
    """把取消检测回调包装为取消时抛出 TaskCancelledException 的回调"""

    async def cancel_cb() -> bool:
        if await cancel_checker():
            raise TaskCancelledException()
        return False

    return cancel_cb
