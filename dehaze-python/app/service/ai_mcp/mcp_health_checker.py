"""外部 MCP Server 健康巡检

Server 的 health 原本仅在管理员手动探测时更新，外部服务自身宕机不会被感知。
本巡检按固定间隔对启用中的 Server 重跑一次探测，刷新 health 与 last_check_time；
单 Server 探测失败只记 warning，不影响主流程。

由 lifespan 在主 Worker 启动（多 Worker 下避免重复探测），与 GPU 指标采集器
同为进程内 asyncio 周期任务——巡检频率低、无跨实例协调需求，不引入 XXL-Job。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session
from app.repository.ai_mcp_server_repository import ai_mcp_server_repository
from app.service.ai_mcp.ai_mcp_server_service import ai_mcp_server_service

logger = logging.getLogger(__name__)

# 巡检间隔（秒）
_CHECK_INTERVAL = 600


class McpHealthChecker:
    """启用中 MCP Server 的周期健康巡检。"""

    def __init__(self, interval: int = _CHECK_INTERVAL) -> None:
        self._interval = interval
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                async with get_db_session() as db:
                    await self.check_all(db)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("MCP Server 健康巡检异常: %s", exc)
            await asyncio.sleep(self._interval)

    async def check_all(self, db: AsyncSession) -> None:
        for server in await ai_mcp_server_repository.list_enabled(db):
            try:
                await ai_mcp_server_service.probe_health(db, server.id)
            except Exception as exc:
                logger.warning("MCP Server %s 巡检探测失败: %s", server.name, exc)


mcp_health_checker = McpHealthChecker()
