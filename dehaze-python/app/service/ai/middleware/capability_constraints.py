"""虚拟文件系统容量限制与任务清单约束中间件

设计文档 §6.1/§6.2（能力扩展）：
- write_file：统计工作区已有内容总大小 + 本次写入，超过 AI_VFS_MAX_BYTES（默认
  100MB）返回错误提示清理或压缩后再写入。
- write_todos：条目数超过 32 返回错误提示合并或精简；单条描述超过 50 字符为
  引导性校验（警告但不强制拦截，放行）。

通过 awrap_tool_call 在工具执行前拦截 write_file / write_todos，无需修改
deepagents 内置工具本身。
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from app.config import settings

logger = logging.getLogger(__name__)

_WRITE_FILE_TOOL = "write_file"
_WRITE_TODOS_TOOL = "write_todos"


def _vfssize(files: dict[str, Any] | None) -> int:
    """统计虚拟文件系统已有内容总字节数。"""
    total = 0
    for data in (files or {}).values():
        if isinstance(data, dict):
            content = data.get("content")
        else:
            content = getattr(data, "content", None)
        if isinstance(content, str):
            total += len(content.encode("utf-8"))
    return total


class CapabilityConstraintsMiddleware(AgentMiddleware):
    """deepagents 内置 write_file / write_todos 的约束包装。"""

    def __init__(self) -> None:
        self.max_bytes = settings.AI_VFS_MAX_BYTES
        self.max_todos = settings.AI_TODOS_MAX_ITEMS
        self.todo_item_max_chars = settings.AI_TODOS_ITEM_MAX_CHARS

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        tool_call = request.tool_call or {}
        name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id", "")

        if name == _WRITE_FILE_TOOL:
            return await self._guard_write_file(request, tool_call, tool_call_id, handler)
        if name == _WRITE_TODOS_TOOL:
            return await self._guard_write_todos(request, tool_call, tool_call_id, handler)
        return await handler(request)

    async def _guard_write_file(
        self,
        request: Any,
        tool_call: dict[str, Any],
        tool_call_id: str,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        args = tool_call.get("args") or {}
        content = args.get("content") or ""
        new_bytes = len(content.encode("utf-8"))
        state = getattr(request, "state", None)
        files = getattr(state, "files", None) if state is not None else None
        if files is None:
            files = state.get("files") if isinstance(state, dict) else {}
        existing = _vfssize(files)
        # 覆盖已存在文件时，原文件内容被替换，先扣除旧内容再判定
        old_bytes = 0
        file_path = args.get("file_path") or ""
        old_data = files.get(file_path) if isinstance(files, dict) else None
        if isinstance(old_data, dict):
            old_bytes = len((old_data.get("content") or "").encode("utf-8"))
        elif old_data is not None:
            old_bytes = len(str(getattr(old_data, "content", "")).encode("utf-8"))
        projected = existing - old_bytes + new_bytes
        if projected > self.max_bytes:
            logger.warning(
                "write_file 容量超限: 现有 %s 字节 + 写入 %s 字节 > 上限 %s 字节",
                existing,
                new_bytes,
                self.max_bytes,
            )
            return ToolMessage(
                content="工作区容量超限，请清理临时文件或压缩后再写入。",
                tool_call_id=tool_call_id,
                status="error",
            )
        return await handler(request)

    async def _guard_write_todos(
        self,
        request: Any,
        tool_call: dict[str, Any],
        tool_call_id: str,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        args = tool_call.get("args") or {}
        todos = args.get("todos") or []
        if len(todos) > self.max_todos:
            return ToolMessage(
                content=f"任务项超过 {self.max_todos} 上限，请合并或精简后再写入。",
                tool_call_id=tool_call_id,
                status="error",
            )
        overlong = [
            str(t.get("content", ""))[:20] + "…"
            for t in todos
            if isinstance(t, dict) and len(str(t.get("content", ""))) > self.todo_item_max_chars
        ]
        if overlong:
            logger.warning(
                "write_todos 存在超长任务描述（>%s 字符），建议精简: %s",
                self.todo_item_max_chars,
                overlong,
            )
        return await handler(request)
