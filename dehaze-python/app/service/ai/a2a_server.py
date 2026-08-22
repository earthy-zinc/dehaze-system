"""A2A 协议服务端（A2AServer）

对外暴露平台 Agent 供外部 Agent 调用：
- GET /.well-known/agent.json：动态生成 Agent Card（由已发布版本生成）
- POST /a2a：JSON-RPC 2.0 入口（message/send / message/stream / tasks/*）

任务存储：
- Task 持久化到 Redis（键 a2a:task:{task_id}，TTL 24h，history/artifacts 序列化 JSON），
  保证多 worker/多实例下 message/send 与 tasks/get 跨进程可见。
- contextId → task 集合索引（a2a:task:ctx:{context_id}，续期式 TTL 25h），
  供 tasks/list 按会话查询；闲置 25h 自动清除，避免残留 stale id。

安全措施：
- API Key 鉴权（Authorization: Bearer dhak_xxx，复用现有 API Key 体系）
- 仅启用且非子 Agent 且 is_exposed=1 的 Agent 可被调用
- 走完整推理链路（护栏/评测/计费照常生效，不因外部来源旁路）
- 评测/外部调用不旁路计费隔离（隔离 Token 池）
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_agent import SysAiAgent
from app.infrastructure.llm.a2a_protocol import (
    Artifact,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    Task,
    parse_part,
)
from app.service.ai.deep_agent_builder import DeepAgentBuilder
from app.service.ai_agent_service import agent_service

logger = logging.getLogger(__name__)

# Task Redis 键前缀与 TTL
_TASK_KEY_PREFIX = "a2a:task:"
_CTX_INDEX_PREFIX = "a2a:task:ctx:"
_TASK_TTL = 24 * 3600


class A2AServer:
    """A2A 协议服务端（单例）"""

    # task_id -> 进行中推理 asyncio.Task（仅存本进程，跨实例 cancel 交给 Redis 状态校验）
    _running: dict[str, asyncio.Task] = {}

    # ── Agent 校验 ─────────────────────────────────────────────

    @staticmethod
    async def _get_exposed_agent(db: AsyncSession, agent_id: int) -> SysAiAgent:
        from app.repository.ai_agent_repository import ai_agent_repository

        agent = await ai_agent_repository.get_by_id(db, agent_id)
        if not agent:
            raise ValueError("Agent 不存在")
        if agent.status != 1 or agent.is_exposed != 1 or agent.is_subagent == 1:
            raise ValueError("Agent 不可对外服务")
        return agent

    # ── Agent Card ─────────────────────────────────────────────

    async def build_agent_card(
        self, db: AsyncSession, redis, agent_id: int, base_url: str
    ) -> dict[str, Any]:
        """由已发布版本动态生成 Agent Card。"""
        agent = await self._get_exposed_agent(db, agent_id)
        snapshot = await agent_service.get_published_snapshot(db, redis, agent_id, None)
        version_no = snapshot.get("version_no", "0.0.0") if snapshot else "0.0.0"
        name = agent.name or agent.agent_code
        skills = [{"name": agent.agent_code, "description": agent.description}]
        return {
            "name": name,
            "description": agent.description,
            "version": f"{version_no}",
            "url": base_url.rstrip("/") + "/a2a",
            "capabilities": {"streaming": True, "pushNotifications": False},
            "defaultInputModes": ["text", "file"],
            "defaultOutputModes": ["text", "file"],
            "skills": skills,
            "securitySchemes": {"http": {"scheme": "bearer"}},
            "security": [{"http": []}],
        }

    # ── JSON-RPC 方法分发 ──────────────────────────────────────

    async def handle(
        self, db: AsyncSession, redis, request: JsonRpcRequest, agent_id: int
    ) -> JsonRpcResponse:
        """分发 JSON-RPC 方法，返回响应（流式方法由调用方单独处理）。"""
        method = request.method
        params = request.params or {}

        if method == "message/send":
            result = await self._message_send(db, redis, agent_id, params)
        elif method == "tasks/get":
            result = await self._task_get(redis, params)
        elif method == "tasks/cancel":
            result = await self._task_cancel(redis, params)
        elif method == "tasks/list":
            result = await self._task_list(redis, params)
        else:
            return JsonRpcResponse(
                id=request.id,
                error=JsonRpcError(code=-32601, message=f"Method not found: {method}"),
            )
        return JsonRpcResponse(id=request.id, result=result)

    # ── message/send ───────────────────────────────────────────

    async def _message_send(self, db: AsyncSession, redis, agent_id: int, params: dict) -> dict:
        task_id = params.get("taskId") or str(uuid.uuid4())
        context_id = params.get("contextId")
        messages = [Message.model_validate(m) for m in params.get("messages", [])]

        agent = await self._get_exposed_agent(db, agent_id)
        snapshot = await agent_service.get_published_snapshot(db, redis, agent_id, None)
        if not snapshot:
            raise ValueError("Agent 无已发布版本")

        await self._register_task(
            redis,
            task_id,
            status="working",
            context_id=context_id,
            history=messages,
            metadata={"agent_id": agent_id},
        )

        # 后台执行推理（不阻塞同步返回，符合异步优先）
        runner = asyncio.create_task(
            self._run_inference(db, redis, agent, snapshot, task_id, messages)
        )
        self._running[task_id] = runner
        runner.add_done_callback(lambda t: self._running.pop(task_id, None))

        task = Task(
            id=task_id,
            context_id=context_id,
            status="working",
            history=messages,
            metadata={"agent_id": agent_id},
        )
        return task.model_dump(by_alias=True, exclude_none=True)

    # ── tasks/* ────────────────────────────────────────────────

    @staticmethod
    async def _task_get(redis, params: dict) -> dict:
        task_id = params.get("taskId", "")
        task = await A2AServer._get_task(redis, task_id)
        if not task:
            raise ValueError("Task 不存在")
        return task

    async def _task_cancel(self, redis, params: dict) -> dict:
        task_id = params.get("taskId", "")
        task = await self._get_task(redis, task_id)
        if not task:
            raise ValueError("Task 不存在")
        # 先落 canceled 状态（跨实例可读），再尝试取消本进程推理
        await self._update_task_status(redis, task_id, "canceled")
        runner = self._running.get(task_id)
        if runner and not runner.done():
            runner.cancel()
        return {"id": task_id, "status": "canceled"}

    @staticmethod
    async def _task_list(redis, params: dict) -> list[dict]:
        context_id = params.get("contextId")
        if context_id:
            index_key = _CTX_INDEX_PREFIX + context_id
            task_ids = await redis.smembers(index_key)
        else:
            # 无 contextId：scan 全部 a2a:task:* 键
            task_ids = []
            async for key in redis.scan_iter(match=_TASK_KEY_PREFIX + "*", count=100):
                task_ids.append(key[len(_TASK_KEY_PREFIX) :])
        tasks = []
        for task_id in task_ids:
            task = await A2AServer._get_task(redis, task_id)
            if task:
                tasks.append(task)
        tasks.sort(key=lambda t: t.get("id", ""))
        return tasks

    # ── 推理执行 ───────────────────────────────────────────────

    async def _run_inference(
        self,
        db: AsyncSession,
        redis,
        agent: SysAiAgent,
        snapshot: dict,
        task_id: str,
        messages: list[Message],
    ) -> None:
        """执行完整推理链路并更新 Task 状态与 Artifact。"""
        try:
            graph = await DeepAgentBuilder().build_from_snapshot(db, redis, snapshot)
            initial_state = {
                "messages": [
                    {"role": m.role, "content": m.to_text()}
                    for m in messages
                    if m.role in ("user", "agent")
                ],
                "user_id": None,
                "conversation_id": 0,
                "message_id": 0,
                "model_id": snapshot.get("model_id", ""),
                "system_prompt": snapshot.get("system_prompt"),
                "stream_session_id": f"a2a:{task_id}",
                "step_count": 0,
                "token_used": 0,
                "token_budget": snapshot.get("config", {}).get("token_budget", 500000),
                "thoughts": [],
                "isolated_token_pool": True,
            }
            config = {"configurable": {"thread_id": f"a2a:{task_id}"}}
            result = await graph.ainvoke(initial_state, config=config)
            final_response = result.get("final_response", "")
            artifacts = []
            if final_response:
                artifacts.append(
                    Artifact(
                        artifact_id=f"{task_id}:output",
                        name="response",
                        parts=[parse_part({"type": "text", "text": final_response})],
                    )
                )
            await self._update_task_status(redis, task_id, "completed", artifacts=artifacts)
        except asyncio.CancelledError:
            await self._update_task_status(redis, task_id, "canceled")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("A2A 任务 %s 推理失败: %s", task_id, exc, exc_info=True)
            await self._update_task_status(redis, task_id, "failed")

    # ── 任务存储（Redis）───────────────────────────────────────

    @staticmethod
    def _task_key(task_id: str) -> str:
        return _TASK_KEY_PREFIX + task_id

    @staticmethod
    def _serialize_task(task: Task, status: str, artifacts: list[Artifact] | None = None) -> dict:
        return {
            "id": task.id,
            "contextId": task.context_id,
            "status": status,
            "artifacts": [
                a.model_dump(by_alias=True, exclude_none=True)
                for a in (artifacts or task.artifacts)
            ],
            "history": [m.model_dump(by_alias=True, exclude_none=True) for m in task.history],
            "metadata": task.metadata,
        }

    @classmethod
    async def _register_task(
        cls,
        redis,
        task_id: str,
        status: str,
        context_id: str | None,
        history: list[Message],
        metadata: dict[str, Any],
    ) -> None:
        """写入 Task 到 Redis，并维护 contextId 集合索引。"""
        task = Task(
            id=task_id, context_id=context_id, status=status, history=history, metadata=metadata
        )
        payload = cls._serialize_task(task, status)
        pipe = redis.pipeline()
        pipe.set(cls._task_key(task_id), json.dumps(payload, ensure_ascii=False), ex=_TASK_TTL)
        if context_id:
            index_key = _CTX_INDEX_PREFIX + context_id
            pipe.sadd(index_key, task_id)
            # 续期式 TTL：活跃会话每次写入续期，闲置 25h 后集合自动清除，
            # 避免 task 键 24h 过期后 contextId 索引残留 stale id。
            pipe.expire(index_key, _TASK_TTL + 3600)
        await pipe.execute()

    @staticmethod
    async def _get_task(redis, task_id: str) -> dict | None:
        raw = await redis.get(A2AServer._task_key(task_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("A2A Task %s 反序列化失败", task_id)
            return None

    @classmethod
    async def _update_task_status(
        cls,
        redis,
        task_id: str,
        status: str,
        artifacts: list[Artifact] | None = None,
    ) -> None:
        """更新 Task 状态（读到现有记录后原地更新 status/artifacts）。"""
        task = await cls._get_task(redis, task_id)
        if not task:
            return
        if artifacts is not None:
            task["artifacts"] = [a.model_dump(by_alias=True, exclude_none=True) for a in artifacts]
        task["status"] = status
        await redis.set(cls._task_key(task_id), json.dumps(task, ensure_ascii=False), ex=_TASK_TTL)


a2a_server = A2AServer()
