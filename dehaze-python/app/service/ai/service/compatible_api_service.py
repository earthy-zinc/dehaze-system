"""第三方兼容 API 服务（F-M08-010）

作为协议适配层（Adapter），复用内部 API 的会话、消息、推理与计费能力：
- 解析 OpenAI Chat Completions / Anthropic Messages 请求
- 会话映射：conversation_id 复用已有会话，否则按首条消息自动创建
- 调用内部 AiMessageService.send_message 驱动同一推理链路
- 将内部 SSE 事件转换为 OpenAI SSE / Claude SSE
- 非流式时聚合事件为单次 JSON 响应
"""

import json
import logging
import time
from uuid import uuid4

from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.models.entity.api_key import SysApiKey
from app.models.schema.ai_conversation import ConversationCreate, MessageSend
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.repository.api_key_repository import api_key_repository
from app.service.ai.service.compatible_audit import record_call
from app.service.ai.service.compatible_governance import (
    compatible_governance_service,
    GovernanceError,
)
from app.service.ai_conversation_service import ai_conversation_service
from app.service.ai_message_service import ai_message_service
from app.service.ai_model_service import ai_model_service

logger = logging.getLogger(__name__)


def _parse_internal_sse(chunk: str) -> tuple[str, dict]:
    """解析内部 SSE 事件文本，返回 (event_type, data)。

    假设内部 SSE 每事件仅一行 data（SseEmitterManager._format_event 对 data 做单行 json.dumps），
    因此不处理 SSE 规范中的多行 data 累积。
    """
    event_type = ""
    data: dict = {}
    for line in chunk.splitlines():
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            try:
                data = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                data = {}
    return event_type, data


def _error_status_code(code: ResultCode) -> int:
    """根据业务错误码映射 HTTP 状态码，对齐官方协议语义（§2.3.1）"""
    if code in (
        ResultCode.TOKEN_INVALID,
        ResultCode.TOKEN_ACCESS_FORBIDDEN,
        ResultCode.ACCESS_UNAUTHORIZED,
        ResultCode.CLIENT_AUTHENTICATION_FAILED,
    ):
        return 401
    # 限流（Key 级配额超限、系统限流）
    if code in (ResultCode.RATE_LIMIT,):
        return 429
    # 用户积分配额不足 → 402 官方"配额不足"语义
    if code in (
        ResultCode.QUOTA_INSUFFICIENT,
        ResultCode.QUOTA_EXCEEDED,
        ResultCode.GROWTH_INSUFFICIENT,
    ):
        return 402
    # 模型不可用 / 无权限 → 403
    if code in (ResultCode.AI_MODEL_NOT_AVAILABLE,):
        return 403
    if code == ResultCode.RESOURCE_NOT_FOUND:
        return 404
    if code in (
        ResultCode.PARAM_ERROR,
        ResultCode.PARAM_IS_NULL,
        ResultCode.USER_ERROR,
        ResultCode.BUSINESS_ERROR,
        ResultCode.DATA_EXISTS,
        ResultCode.DATA_STATE_NOT_ALLOW,
        ResultCode.OPERATION_NOT_ALLOW,
        ResultCode.DATA_BIND_EXISTS,
        ResultCode.AI_LLM_CALL_FAILED,
    ):
        return 400
    return 500


def _error_type_for(code: ResultCode) -> str:
    """业务错误码 → 官方错误 type（OpenAI/Claude 协议共用）"""
    if _error_status_code(code) == 429:
        return "rate_limit_error"
    if _error_status_code(code) == 403:
        return "permission_error"
    if _error_status_code(code) == 402:
        return "insufficient_quota"
    return "invalid_request_error"


def _openai_error(message: str, error_type: str, code: str | None = None) -> dict:
    """OpenAI 协议错误响应体"""
    return {"error": {"message": message, "type": error_type, "code": code}}


def _claude_error(message: str, error_type: str) -> dict:
    """Claude 协议错误响应体"""
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _format_error(protocol: str, message: str, error_type: str, code: str | None = None) -> dict:
    """按协议返回统一错误响应体"""
    if protocol == "claude":
        return _claude_error(message, error_type)
    return _openai_error(message, error_type, code)


def _to_int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_audit(
    audit: dict,
    *,
    status_code: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    credits: float | None = None,
    error_msg: str | None = None,
) -> None:
    """统一审计写入（fire-and-forget，绝不外抛）"""
    try:
        duration_ms = int((time.monotonic() - audit.get("t0", time.monotonic())) * 1000)
        record_call(
            user_id=audit.get("user_id"),
            key_id=audit.get("key_id"),
            key_prefix=audit.get("key_prefix", ""),
            conversation_id=audit.get("conversation_id"),
            model=audit.get("model"),
            endpoint=audit.get("endpoint", ""),
            protocol=audit.get("protocol", ""),
            is_stream=bool(audit.get("is_stream", False)),
            status_code=status_code,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            credits=credits,
            error_msg=error_msg,
            request_id=audit.get("request_id"),
            client_ip=audit.get("client_ip", ""),
            duration_ms=duration_ms,
        )
    except Exception:  # noqa: BLE001 - 审计失败不影响主流程
        logger.warning("兼容 API 调用审计记录失败", exc_info=True)


def _openai_usage(usage: dict) -> dict:
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _claude_usage(usage: dict) -> dict:
    return {
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
    }


def _openai_finish_reason(stop_reason: str) -> str:
    return {
        "stop": "stop",
        "tool_calls": "tool_calls",
        "length": "length",
        "content_filter": "content_filter",
    }.get(stop_reason, "stop")


def _claude_stop_reason(stop_reason: str) -> str:
    return {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }.get(stop_reason, "end_turn")


def _openai_sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _claude_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_system_prompt(messages: list[dict]) -> str | None:
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            return msg["content"]
    return None


def _extract_last_user_content(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # OpenAI 多模态格式：拼接所有 text 块；image_url 等图片块暂不支持兼容 API 多模态，忽略
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if parts:
                return "\n".join(parts)
    return ""


async def _openai_stream(
    internal_response, model: str, message_id: str, created: int, audit: dict | None = None
):
    """将内部 SSE 事件转换为 OpenAI Chat Completions 流式格式"""
    base = {"id": message_id, "object": "chat.completion.chunk", "created": created, "model": model}
    usage: dict = {}
    error_msg: str | None = None
    try:
        yield _openai_sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
        )
        async for chunk in internal_response.body_iterator:
            event_type, data = _parse_internal_sse(chunk)
            if event_type == "content_block.delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    yield _openai_sse(
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta.get("text", "")},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                elif delta.get("type") == "input_json_delta":
                    yield _openai_sse(
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": data.get("index", 0),
                                                "function": {
                                                    "arguments": delta.get("partial_json", "")
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            elif event_type == "content_block.start":
                if data.get("type") == "tool_use":
                    yield _openai_sse(
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": data.get("index", 0),
                                                "id": data.get("id", ""),
                                                "type": "function",
                                                "function": {
                                                    "name": data.get("name", ""),
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            elif event_type == "message.end":
                usage = data.get("usage") or {}
                yield _openai_sse(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": _openai_finish_reason(
                                    data.get("stopReason", "stop")
                                ),
                            }
                        ],
                        "usage": _openai_usage(usage),
                    }
                )
                yield "data: [DONE]\n\n"
            elif event_type == "error":
                error_msg = data.get("message") or "推理失败"
                yield _openai_sse(
                    {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                )
                yield "data: [DONE]\n\n"
    finally:
        if audit is not None:
            _record_audit(
                audit,
                status_code=500 if error_msg else 200,
                input_tokens=usage.get("inputTokens", 0),
                output_tokens=usage.get("outputTokens", 0),
                credits=usage.get("credits"),
                error_msg=error_msg,
            )


async def _openai_non_stream(
    internal_response, model: str, message_id: str, created: int, audit: dict | None = None
) -> dict:
    """聚合内部 SSE 事件为 OpenAI 非流式 JSON 响应"""
    content = ""
    tool_calls: list[dict] = []
    usage: dict = {}
    stop_reason = "stop"
    error_msg: str | None = None
    async for chunk in internal_response.body_iterator:
        event_type, data = _parse_internal_sse(chunk)
        if event_type == "content_block.delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                content += delta.get("text", "")
        elif event_type == "content_block.start":
            if data.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": data.get("id", ""),
                        "type": "function",
                        "function": {"name": data.get("name", ""), "arguments": ""},
                    }
                )
        elif event_type == "message.end":
            stop_reason = data.get("stopReason", "stop")
            usage = data.get("usage") or {}
        elif event_type == "error":
            error_msg = data.get("message") or "推理失败"
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if audit is not None:
        _record_audit(
            audit,
            status_code=500 if error_msg else 200,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            credits=usage.get("credits"),
            error_msg=error_msg,
        )
    return {
        "id": message_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": _openai_finish_reason(stop_reason)}
        ],
        "usage": _openai_usage(usage),
    }


async def _claude_stream(internal_response, model: str, message_id: str, audit: dict | None = None):
    """将内部 SSE 事件转换为 Anthropic Messages 流式格式"""
    usage: dict = {}
    error_msg: str | None = None
    try:
        yield _claude_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        async for chunk in internal_response.body_iterator:
            event_type, data = _parse_internal_sse(chunk)
            if event_type == "content_block.delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    yield _claude_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": data.get("index", 0),
                            "delta": {"type": "text_delta", "text": delta.get("text", "")},
                        },
                    )
                elif delta.get("type") == "input_json_delta":
                    yield _claude_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": data.get("index", 0),
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": delta.get("partial_json", ""),
                            },
                        },
                    )
            elif event_type == "content_block.start":
                if data.get("type") == "tool_use":
                    yield _claude_sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": data.get("index", 0),
                            "content_block": {
                                "type": "tool_use",
                                "id": data.get("id", ""),
                                "name": data.get("name", ""),
                                "input": {},
                            },
                        },
                    )
            elif event_type == "content_block.stop":
                yield _claude_sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": data.get("index", 0)},
                )
            elif event_type == "message.end":
                usage = data.get("usage") or {}
                yield _claude_sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": _claude_stop_reason(data.get("stopReason", "stop")),
                            "stop_sequence": None,
                        },
                        "usage": _claude_usage(usage),
                    },
                )
                yield _claude_sse("message_stop", {"type": "message_stop"})
            elif event_type == "error":
                error_msg = data.get("message") or "推理失败"
                yield _claude_sse(
                    "error",
                    {
                        "type": "error",
                        "error": {"type": "api_error", "message": error_msg},
                    },
                )
    finally:
        if audit is not None:
            _record_audit(
                audit,
                status_code=500 if error_msg else 200,
                input_tokens=usage.get("inputTokens", 0),
                output_tokens=usage.get("outputTokens", 0),
                credits=usage.get("credits"),
                error_msg=error_msg,
            )


async def _claude_non_stream(
    internal_response, model: str, message_id: str, audit: dict | None = None
) -> dict:
    """聚合内部 SSE 事件为 Anthropic 非流式 JSON 响应"""
    text = ""
    content: list[dict] = []
    usage: dict = {}
    stop_reason = "stop"
    error_msg: str | None = None
    async for chunk in internal_response.body_iterator:
        event_type, data = _parse_internal_sse(chunk)
        if event_type == "content_block.delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                text += delta.get("text", "")
        elif event_type == "content_block.start":
            if data.get("type") == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": data.get("id", ""),
                        "name": data.get("name", ""),
                        "input": {},
                    }
                )
        elif event_type == "message.end":
            stop_reason = data.get("stopReason", "stop")
            usage = data.get("usage") or {}
        elif event_type == "error":
            error_msg = data.get("message") or "推理失败"
    if text:
        content.insert(0, {"type": "text", "text": text})
    if audit is not None:
        _record_audit(
            audit,
            status_code=500 if error_msg else 200,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            credits=usage.get("credits"),
            error_msg=error_msg,
        )
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _claude_stop_reason(stop_reason),
        "stop_sequence": None,
        "usage": _claude_usage(usage),
    }


class CompatibleApiService:
    """OpenAI / Claude 兼容协议适配服务"""

    def __init__(
        self,
        api_key_repository=api_key_repository,
        ai_conversation_repository=ai_conversation_repository,
    ):
        self.api_key_repository = api_key_repository
        self.ai_conversation_repository = ai_conversation_repository

    async def _get_api_key(self, db, key_id: int | None) -> SysApiKey | None:
        """按 key_id 查询 SysApiKey（治理预检用），不存在返回 None"""
        if not key_id:
            return None
        return await self.api_key_repository.get_by_id(db, key_id)

    async def _resolve_conversation(
        self,
        db,
        user_id: int,
        conversation_id,
        model: str | None,
        system_prompt: str | None,
        first_user_content: str,
    ) -> int:
        """会话映射：conversation_id 复用已有会话，否则按首条消息自动创建"""
        if conversation_id:
            try:
                conversation_id = int(conversation_id)
            except (TypeError, ValueError):
                raise BusinessException(
                    ResultCode.PARAM_ERROR, "conversation_id 必须是整数"
                ) from None
            conv = await self.ai_conversation_repository.get_by_id_and_user(
                db, conversation_id, user_id
            )
            if not conv:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
            return conv.id
        title = first_user_content[:50] if first_user_content else "新对话"
        conv = await ai_conversation_service.create_conversation(
            db, user_id, ConversationCreate(title=title, model=model, systemPrompt=system_prompt)
        )
        return conv.id

    async def _enforce_model_whitelist(
        self, db, user_id: int, api_key, model: str | None, conv_id: int | None
    ) -> None:
        """会话默认模型场景的二次白名单校验（模型白名单见 §2.3）。

        请求未显式指定模型时走会话默认模型（对齐 send_message 的解析逻辑），
        该模型在 precheck（仅校验显式 model）之后才确定，需在此兜底校验。
        """
        if api_key is None or model is not None:
            return
        used_model = settings.AI_DEFAULT_MODEL
        if conv_id is not None:
            convs = await self.ai_conversation_repository.get_by_ids(db, user_id, [conv_id])
            if convs and convs[0].model:
                used_model = convs[0].model
        await compatible_governance_service.check_model_allowed(db, api_key, used_model)

    async def handle_openai_chat(
        self, db, user_id: int, body: dict, api_key=None, audit: dict | None = None
    ):
        """处理 OpenAI Chat Completions 兼容请求"""
        messages = body.get("messages") or []
        if not messages:
            raise BusinessException(ResultCode.PARAM_ERROR, "messages 不能为空")
        user_content = _extract_last_user_content(messages)
        if not user_content:
            raise BusinessException(ResultCode.PARAM_ERROR, "缺少用户消息")
        if len(user_content) > settings.AI_MESSAGE_MAX_LENGTH:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, f"消息长度不能超过 {settings.AI_MESSAGE_MAX_LENGTH} 字符"
            )
        model = body.get("model")
        stream = bool(body.get("stream", False))
        conv_id = await self._resolve_conversation(
            db,
            user_id,
            body.get("conversation_id"),
            model,
            _extract_system_prompt(messages),
            user_content,
        )
        if audit is not None:
            audit["conversation_id"] = conv_id
        await self._enforce_model_whitelist(db, user_id, api_key, model, conv_id)
        internal_response = await ai_message_service.send_message(
            db, conv_id, user_id, MessageSend(content=user_content, model=model), str(uuid4())
        )
        message_id = "chatcmpl-" + uuid4().hex
        created = int(time.time())
        if stream:
            return StreamingResponse(
                _openai_stream(internal_response, model or "", message_id, created, audit),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return JSONResponse(
            content=await _openai_non_stream(
                internal_response, model or "", message_id, created, audit
            )
        )

    async def handle_claude_messages(
        self, db, user_id: int, body: dict, api_key=None, audit: dict | None = None
    ):
        """处理 Anthropic Messages 兼容请求"""
        messages = body.get("messages") or []
        if not messages:
            raise BusinessException(ResultCode.PARAM_ERROR, "messages 不能为空")
        user_content = _extract_last_user_content(messages)
        if not user_content:
            raise BusinessException(ResultCode.PARAM_ERROR, "缺少用户消息")
        if len(user_content) > settings.AI_MESSAGE_MAX_LENGTH:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, f"消息长度不能超过 {settings.AI_MESSAGE_MAX_LENGTH} 字符"
            )
        model = body.get("model")
        stream = bool(body.get("stream", False))
        conv_id = await self._resolve_conversation(
            db, user_id, body.get("conversation_id"), model, body.get("system"), user_content
        )
        if audit is not None:
            audit["conversation_id"] = conv_id
        await self._enforce_model_whitelist(db, user_id, api_key, model, conv_id)
        internal_response = await ai_message_service.send_message(
            db, conv_id, user_id, MessageSend(content=user_content, model=model), str(uuid4())
        )
        message_id = "msg_" + uuid4().hex
        if stream:
            return StreamingResponse(
                _claude_stream(internal_response, model or "", message_id, audit),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return JSONResponse(
            content=await _claude_non_stream(internal_response, model or "", message_id, audit)
        )

    @staticmethod
    async def list_models_openai(db, redis, user_id: int, api_key: SysApiKey | None = None) -> dict:
        """模型列表（OpenAI 格式）。

        与内部 API 一致的用户 VIP 过滤（AiModelService.list_enabled_models，
        含缓存），再叠加 Key 白名单过滤（§2.3 模型白名单）。
        """
        models = await compatible_governance_service.filter_models(
            db, api_key, await ai_model_service.list_enabled_models(db, redis, user_id)
        )
        return {
            "object": "list",
            "data": [
                {
                    "id": m.model_id,
                    "object": "model",
                    "created": int(m.create_time.timestamp()) if m.create_time else 0,
                    "owned_by": m.provider_id,
                }
                for m in models
            ],
        }

    @staticmethod
    async def list_models_claude(db, redis, user_id: int, api_key: SysApiKey | None = None) -> dict:
        """模型列表（Claude 格式）。

        与内部 API 一致的用户 VIP 过滤（AiModelService.list_enabled_models，
        含缓存），再叠加 Key 白名单过滤（§2.3 模型白名单）。
        """
        models = await compatible_governance_service.filter_models(
            db, api_key, await ai_model_service.list_enabled_models(db, redis, user_id)
        )
        data = [
            {
                "type": "model",
                "id": m.model_id,
                "display_name": m.display_name,
                "created_at": m.create_time.isoformat() if m.create_time else None,
            }
            for m in models
        ]
        return {
            "data": data,
            "has_more": False,
            # 当前不支持分页，无更多数据时游标应为空
            "first_id": None,
            "last_id": None,
        }


    async def run_compatible_call(
        self,
        request,
        db,
        user,
        *,
        protocol: str,
        endpoint: str,
        handler,
    ):
        """兼容端点统一入口：治理预检 + 业务执行 + 统一错误格式化 + 全路径审计埋点。

        - handler(body, audit, api_key) 由各协议路由注入，返回 StreamingResponse 或 JSONResponse；
          成功路径（流式/非流式）的审计由 service 侧写入，此处仅处理异常路径。
        - API Key 认证（request.state.api_key_info 存在）时执行 Key 级治理预检；
          Session 身份访问时跳过治理（Key 级配额/白名单不适用），审计 key_id=None。
        """
        t0 = time.monotonic()
        api_key_info = getattr(request.state, "api_key_info", None)
        api_key = None
        key_id = None
        key_prefix = ""
        if api_key_info:
            key_id = api_key_info.get("key_id")
            key_prefix = api_key_info.get("key_prefix", "") or ""
            api_key = await self._get_api_key(db, key_id)
        audit = {
            "user_id": user.id if user else None,
            "key_id": api_key.id if api_key else key_id,
            "key_prefix": key_prefix,
            "endpoint": endpoint,
            "protocol": protocol,
            "client_ip": request.client.host if request.client else "",
            "request_id": getattr(request.state, "request_id", None),
            "t0": t0,
            "conversation_id": None,
            "model": None,
            "is_stream": False,
        }
        try:
            body = await request.json()
            audit["model"] = body.get("model")
            audit["is_stream"] = bool(body.get("stream", False))
            audit["conversation_id"] = _to_int_or_none(body.get("conversation_id"))
            # Key 级治理预检：仅 API Key 认证路径；model=None 时白名单留待会话默认模型二次校验
            if api_key is not None:
                redis = await get_redis_client()
                await compatible_governance_service.precheck(
                    redis, api_key, audit["model"], endpoint
                )
            return await handler(body, audit, api_key)
        except GovernanceError as e:
            _record_audit(audit, status_code=e.status_code, error_msg=e.message)
            return JSONResponse(
                status_code=e.status_code,
                content=_format_error(protocol, e.message, e.error_type),
            )
        except BusinessException as e:
            status_code = _error_status_code(e.code)
            _record_audit(audit, status_code=status_code, error_msg=e.message)
            return JSONResponse(
                status_code=status_code,
                content=_format_error(protocol, e.message, _error_type_for(e.code), e.code.code),
            )
        except Exception as e:  # noqa: BLE001 - 兜底保证审计不遗漏，映射为 500
            logger.exception("兼容 API 处理失败 protocol=%s endpoint=%s", protocol, endpoint)
            _record_audit(audit, status_code=500, error_msg=str(e))
            return JSONResponse(
                status_code=500,
                content=_format_error(protocol, "服务器内部错误", "server_error"),
            )

    async def run_models_call(
        self,
        request,
        db,
        user,
        *,
        protocol: str,
        handler,
    ):
        """模型列表端点统一入口：治理预检 + 白名单过滤 + 审计（对齐 §2.3.1 endpoint 枚举含 models）。

        models 为轻量 GET（无 body/无 token 消耗），但仍纳入 Key 级配额/RPM 计数——
        否则狂刷 models 可绕过治理；调用记录 tokens=0、model=None。
        Session 身份访问时跳过治理（对齐对话端点语义），审计 key_id=None。
        """
        t0 = time.monotonic()
        api_key_info = getattr(request.state, "api_key_info", None)
        api_key = None
        if api_key_info:
            api_key = await self._get_api_key(db, api_key_info.get("key_id"))
        audit = {
            "user_id": user.id if user else None,
            "key_id": api_key.id if api_key else api_key_info.get("key_id") if api_key_info else None,
            "key_prefix": (api_key_info.get("key_prefix", "") or "") if api_key_info else "",
            "endpoint": "models",
            "protocol": protocol,
            "client_ip": request.client.host if request.client else "",
            "request_id": getattr(request.state, "request_id", None),
            "t0": t0,
            "conversation_id": None,
            "model": None,
            "is_stream": False,
        }
        try:
            redis = await get_redis_client()
            if api_key is not None:
                await compatible_governance_service.precheck(redis, api_key, None, "models")
            data = await handler(db, redis, user.id, api_key)
            _record_audit(audit, status_code=200)
            return JSONResponse(status_code=200, content=data)
        except GovernanceError as e:
            _record_audit(audit, status_code=e.status_code, error_msg=e.message)
            return JSONResponse(
                status_code=e.status_code,
                content=_format_error(protocol, e.message, e.error_type),
            )
        except BusinessException as e:
            status_code = _error_status_code(e.code)
            _record_audit(audit, status_code=status_code, error_msg=e.message)
            return JSONResponse(
                status_code=status_code,
                content=_format_error(protocol, e.message, _error_type_for(e.code), e.code.code),
            )
        except Exception as e:  # noqa: BLE001 - 兜底保证审计不遗漏，映射为 500
            logger.exception("兼容 API 模型列表处理失败 protocol=%s", protocol)
            _record_audit(audit, status_code=500, error_msg=str(e))
            return JSONResponse(
                status_code=500,
                content=_format_error(protocol, "服务器内部错误", "server_error"),
            )


compatible_api_service = CompatibleApiService()
