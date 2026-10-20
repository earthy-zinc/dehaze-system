"""外部 MCP Server 健康巡检

Server 的 health 原本仅在管理员手动探测时更新，外部服务自身宕机不会被感知。
本巡检按固定间隔对启用中的 Server 重跑一次探测，刷新 health 与 last_check_time；
单 Server 探测失败只记 warning，不影响主流程。

由 lifespan 在主 Worker 启动（多 Worker 下避免重复探测），与 GPU 指标采集器
同为进程内 asyncio 周期任务——巡检频率低、无跨实例协调需求，不引入 XXL-Job。

单轮先只读一次启用中 Server 的 id，随后每个 Server 各用独立短事务探测、并发受
concurrency 限制：探测是外部 HTTP（最长 5s），若整轮共用一个事务，一轮的耗时
会全部压在该事务上——长时间占住数据库连接，并把 SLOW_SQL 记成巡检语句本身。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from app.database import get_db_session
from app.repository.ai_mcp_server_repository import ai_mcp_server_repository
from app.service.ai_mcp.ai_mcp_server_service import ai_mcp_server_service

logger = logging.getLogger(__name__)

# 巡检间隔（秒）
_CHECK_INTERVAL = 600
# 单轮并发探测数上限：探测为外部 IO 密集，并发可把一轮从数十秒压到数秒
_CHECK_CONCURRENCY = 8


class McpHealthChecker:
    """启用中 MCP Server 的周期健康巡检。"""

    def __init__(
        self,
        interval: int = _CHECK_INTERVAL,
        concurrency: int = _CHECK_CONCURRENCY,
    ) -> None:
        self._interval = interval
        self._concurrency = concurrency
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
            started = time.monotonic()
            try:
                await self.check_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("MCP Server 健康巡检异常: %s", exc)
            # 固定节拍：轮次耗时从间隔里扣除，周期不随探测耗时漂移
            await asyncio.sleep(max(0.0, self._interval - (time.monotonic() - started)))

    async def check_all(self) -> None:
        """探测全部启用中 Server：每个 Server 各自一个短事务，探测期间不与其他 Server 共事务。"""
        async with get_db_session() as db:
            server_ids = [server.id for server in await ai_mcp_server_repository.list_enabled(db)]
        if not server_ids:
            return

        semaphore = asyncio.Semaphore(self._concurrency)

        async def probe(server_id: int) -> None:
            async with semaphore:
                try:
                    async with get_db_session() as db:
                        await ai_mcp_server_service.probe_health(db, server_id)
                except Exception as exc:
                    logger.warning("MCP Server %s 巡检探测失败: %s", server_id, exc)

        await asyncio.gather(*(probe(server_id) for server_id in server_ids))


mcp_health_checker = McpHealthChecker()
