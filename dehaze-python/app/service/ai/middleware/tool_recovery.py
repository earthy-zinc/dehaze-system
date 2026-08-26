"""工具调用错误分类与恢复策略（设计文档 §6.4 错误恢复）

工具调用失败时按错误类型执行系统化恢复，恢复动作由 after_tool 钩子产出、
DehazeHooksMiddleware.awrap_tool_call 应用：

| 错误类型 | 判定 | 恢复动作 |
|---------|------|---------|
| 参数错误 | ValueError / 工具自定义参数语义错误 | 重试：把错误信息作为 ToolMessage 返回 LLM |
|         |                                     | 修正参数（retry_max 上限）                     |
| 超时     | asyncio.TimeoutError | 失败：标记超时（降并行语义），上限后不再重试 |
| 权限不足 | PermissionError / 自定义授权异常 | 中断：interrupt(type=confirm) 请求用户授权 |
| 服务不可用 | httpx 5xx / 连接错误 | 跳过：记录为 skipped，继续后续步骤 |
| 不可恢复 | 其他异常 | 失败：记录错误并生成含失败信息的回复 |

thought 的 status 约定：1 成功 / 2 失败 / 3 跳过，error 记录失败原因（§6.4.1 错误透明）。
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

# 工具返回值中标识错误类型的关键字（工具自定义错误语义，缺省归类为参数错误）
_PARAM_ERROR_MARKERS = ("参数", "parameter", "argument", "invalid", "schema", "validation")


@dataclass
class RecoveryAction:
    """错误恢复动作。

    action: retry / fail / interrupt / skip
    reason: 失败/跳过原因（写入 thought.error）
    status: 2 失败 / 3 跳过（对 thought 落库透出）
    """

    action: str
    reason: str
    status: int = 2


def classify_tool_error(exc: Exception) -> RecoveryAction:
    """将工具调用异常归类为恢复动作（§6.4 错误分类）。

    分类优先级：权限不足 > 超时 > 服务不可用 > 参数错误 > 不可恢复。
    """
    if isinstance(exc, PermissionError):
        return RecoveryAction(
            action="interrupt",
            reason="权限不足，需用户确认授权",
            status=2,
        )
    if isinstance(exc, asyncio.TimeoutError):
        return RecoveryAction(
            action="fail",
            reason="工具调用超时",
            status=2,
        )
    if isinstance(exc, httpx.HTTPStatusError) and 500 <= exc.response.status_code <= 599:
        return RecoveryAction(
            action="skip",
            reason=f"下游服务不可用: HTTP {exc.response.status_code}",
            status=3,
        )
    if isinstance(exc, (httpx.ConnectError, httpx.TransportError)):
        return RecoveryAction(
            action="skip",
            reason="下游服务连接失败",
            status=3,
        )
    if isinstance(exc, ValueError):
        return RecoveryAction(
            action="retry",
            reason=f"工具参数错误: {exc}",
            status=2,
        )
    return RecoveryAction(
        action="fail",
        reason=f"不可恢复错误: {exc}",
        status=2,
    )


def is_param_error_message(content: Any) -> bool:
    """判断工具返回内容是否为参数错误（用于重试路径的错误识别）。"""
    if not isinstance(content, str):
        return False
    lowered = content.lower()
    return any(marker in lowered for marker in _PARAM_ERROR_MARKERS)
