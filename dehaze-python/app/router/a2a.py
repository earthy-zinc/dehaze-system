"""A2A 协议路由（AI对话 - 智能体管理）

对外暴露 Agent 供外部 Agent 调用：
- GET  {agent}/a2a/.well-known/agent.json  动态生成 Agent Card（发现端点）
- POST {agent}/a2a                          JSON-RPC 2.0 入口（message/send 等）

认证：Authorization: Bearer dhak_xxx（由 ApiKeyAuthMiddleware 校验，
请求头 x-api-key 兼容 Anthropic 协议）。
"""

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessException
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis_client
from app.service.ai.a2a_protocol import (
    Artifact,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    parse_part,
)
from app.service.ai.a2a_server import a2a_server
from app.service.ai.deep_agent_builder import DeepAgentBuilder
from app.service.ai_agent_service import agent_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai/agents/{agent_id}/a2a", tags=["AI对话-A2A协议"])

# JSON-RPC 错误码
_ERR_PARSE = -32700
_ERR_INVALID_REQUEST = -32600
_ERR_INTERNAL = -32603


@router.get("/.well-known/agent.json", summary="获取 Agent Card（动态生成）")
async def agent_card(
    agent_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis_client),
    user: UserContext = Depends(get_current_user),
):
    base_url = str(request.base_url).rstrip("/")
    card = await a2a_server.build_agent_card(db, redis, agent_id, base_url)
    return JSONResponse(content=card)


@router.post("", summary="A2A JSON-RPC 入口")
async def a2a_entry(
    agent_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis_client),
    user: UserContext = Depends(get_current_user),
):
    try:
        raw = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            status_code=400,
            content=JsonRpcError(code=_ERR_PARSE, message="Invalid JSON").model_dump(),
        )

    try:
        rpc = JsonRpcRequest.model_validate(raw)
    except Exception:
        return JSONResponse(
            status_code=400,
            content=JsonRpcError(
                code=_ERR_INVALID_REQUEST, message="Invalid JSON-RPC request"
            ).model_dump(),
        )

    if rpc.method == "message/stream":
        return StreamingResponse(
            _stream_message(agent_id, db, redis, rpc),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        resp = await a2a_server.handle(db, redis, rpc, agent_id)
        status_code = 400 if resp.error and resp.error.code < -32000 else 200
        return JSONResponse(status_code=status_code, content=resp.model_dump(exclude_none=True))
    except ValueError as e:
        return JSONResponse(
            status_code=404,
            content=JsonRpcResponse(
                id=rpc.id, error=JsonRpcError(code=_ERR_INTERNAL, message=str(e))
            ).model_dump(exclude_none=True),
        )
    except Exception as e:  # noqa: BLE001
        logger.error("A2A 请求处理失败: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content=JsonRpcResponse(
                id=rpc.id, error=JsonRpcError(code=_ERR_INTERNAL, message="Internal error")
            ).model_dump(exclude_none=True),
        )


async def _stream_message(
    agent_id: int,
    db: AsyncSession,
    redis,
    rpc: JsonRpcRequest,
) -> AsyncGenerator[str, None]:
    """message/stream：SSE 流式返回任务进展事件。"""
    params = rpc.params or {}
    task_id = params.get("taskId") or f"a2a:{id(rpc)}"
    messages = [Message.model_validate(m) for m in params.get("messages", [])]

    try:
        await a2a_server._get_exposed_agent(db, agent_id)
        snapshot = await agent_service.get_published_snapshot(db, redis, agent_id, None)
        if not snapshot:
            raise BusinessException("Agent 无已发布版本")

        await a2a_server._register_task(
            redis,
            task_id,
            status="submitted",
            context_id=None,
            history=messages,
            metadata={"agent_id": agent_id},
        )
        yield _sse("status-update", {"id": task_id, "status": "submitted"})
        yield _sse("status-update", {"id": task_id, "status": "working"})

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
        for art in artifacts:
            await a2a_server._update_task_status(redis, task_id, "working", artifacts=[art])
            yield _sse(
                "artifact-update",
                {
                    "id": task_id,
                    "artifact": art.model_dump(by_alias=True, exclude_none=True),
                },
            )
        await a2a_server._update_task_status(redis, task_id, "completed", artifacts=artifacts)
        yield _sse("status-update", {"id": task_id, "status": "completed"})
    except Exception as e:  # noqa: BLE001
        logger.error("A2A 流式推理失败: %s", e, exc_info=True)
        await a2a_server._update_task_status(redis, task_id, "failed")
        yield _sse("status-update", {"id": task_id, "status": "failed"})
        yield _sse("error", {"message": str(e)})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
