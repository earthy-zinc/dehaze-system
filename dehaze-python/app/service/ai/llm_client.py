"""LLM 客户端（带韧性链路的调用客户端）

根据 model_id 从数据库构建「候选路由序列」，逐候选路由、逐 API Key 重试调用：

- 候选路由序列（顺序）：当前 (model, provider) → 同模型备用供应商 → 降级链各级
- 每个候选路由内部：按 Key 优先级组逐 Key 尝试
  - 调用失败（401/403/429/5xx/超时/连接错误）→ 标记 Key 失败 → 切换下一 Key
  - Key 耗尽 → 记录供应商调用失败 → 下一候选路由
- 全部候选失败 → 抛业务异常「主模型和降级模型均不可用」

流式调用的失败分两段：
- 连接/首字节前失败：可切换 Key / 候选路由重试
- 流中断（已下发部分内容）：标记 Key 失败后抛出，不重试整个请求（无法重放）

调用成功后透出实际使用的 model/provider/key/latency/request_id 给调用方
（供 dehaze_chat_model → agent_hooks 计费归因使用）。

Prompt Caching：anthropic 协议按 prompt_cache_prefix_len 对稳定前缀
（system + 工具定义）注入 cache_control；openai_compat 自动缓存无需干预。
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

import httpx

from app.config import settings
from app.service.ai.local_llm_manager import ensure_running
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import _get_trace_id
from app.infrastructure.crypto.aes_cipher import decrypt
from app.models.base import get_current_user_id
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_provider_repository import ai_provider_repository
from app.service.ai.provider_health_service import provider_health_service
from app.service.ai_model_service import AiModelService
from app.service.ai_provider_key_service import AiProviderKeyService

logger = logging.getLogger(__name__)

# 调用失败错误码
_ERROR_5XX = "5xx"


@dataclass
class LlmStreamChunk:
    """统一的 LLM 流式响应块"""

    # type: text_delta / thinking_delta / tool_call_start / tool_call_delta / tool_call_complete / done
    type: str
    content: str = ""
    usage: dict | None = None
    tool_call_id: str = ""
    tool_call_name: str = ""


class _RouteFailed(Exception):
    """某候选路由（供应商）全部 Key 调用失败，用于切换下一候选路由"""

    def __init__(self, error_code: str, detail: str) -> None:
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail


def _map_httpx_error(exc: Exception) -> tuple[str, str]:
    """将 httpx 异常映射为 (error_code, detail)"""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if 500 <= status <= 599:
            return _ERROR_5XX, f"供应商服务端错误: HTTP {status}"
        return str(status), f"供应商返回 HTTP {status}"
    if isinstance(exc, httpx.ConnectError):
        return "connection", "供应商连接失败"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", "供应商请求超时"
    if isinstance(exc, httpx.TransportError):
        return "transport", "供应商传输错误"
    return "unknown", str(exc)


def build_auth_headers(provider, api_key: str) -> dict:
    """按供应商认证方式构建请求头（合并 default_headers）。

    LlmClient 与连通性测试共用，认证头组装单一实现。
    """
    headers = dict(provider.default_headers or {})
    if provider.auth_type == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"
    elif provider.auth_type == "x-api-key":
        headers["x-api-key"] = api_key
    elif provider.auth_type == "custom":
        # 自定义认证头：头名在 default_headers 中以 auth_header 键配置
        header_name = headers.pop("auth_header", "Authorization")
        headers[header_name] = api_key
    return headers


class LlmClient:
    """LLM 客户端（单例）"""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(settings.AI_MESSAGE_STREAM_TIMEOUT))

    @staticmethod
    def _convert_messages(messages: list[dict], system_prompt: str | None) -> list[dict]:
        """将内部消息列表转换为 OpenAI 兼容格式。

        消息 dict 携带 role/content，并可能携带 tool_call_id（role=tool）
        或 tool_calls（role=assistant），直接透传即可。
        """
        converted = []
        if system_prompt:
            converted.append({"role": "system", "content": system_prompt})
        converted.extend(messages)
        return converted

    @staticmethod
    def _convert_messages_anthropic(messages: list[dict]) -> list[dict]:
        """将内部消息列表转换为 Anthropic 原生格式。

        - role=tool → role=user + tool_result 内容块
        - role=assistant 携带 tool_calls → 追加 tool_use 内容块
        """
        converted = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": msg.get("content", ""),
                            }
                        ],
                    }
                )
            elif role == "assistant":
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    try:
                        input_obj = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        input_obj = {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": input_obj,
                        }
                    )
                if content:
                    converted.append({
                        "role": "assistant", 
                        "content": content
                    })
            else:
                converted.append({
                    "role": role, 
                    "content": msg.get("content", "")
                })
        return converted

    @staticmethod
    def _convert_tools_anthropic(tools: list[dict]) -> list[dict]:
        """将 OpenAI Function 工具定义转换为 Anthropic 原生格式"""
        converted = []
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            converted.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {
                        "type": "object", 
                        "properties": {}
                    }),
                }
            )
        return converted

    @staticmethod
    def _convert_tool_choice_anthropic(tool_choice: str | None) -> dict | None:
        """将 OpenAI tool_choice 字符串转换为 Anthropic 原生格式"""
        if tool_choice is None:
            return None
        mapping = {
            "auto": "auto", 
            "none": "none", 
            "required": "any", 
            "any": "any"
        }
        if tool_choice in mapping:
            return {"type": mapping[tool_choice]}
        # 指定具体工具名
        return {"type": "tool", "name": tool_choice}

    def _should_cache(self, model) -> bool:
        """是否启用 Prompt Caching（anthropic 需主动注入 cache_control）"""
        return bool(model.supports_prompt_cache and model.prompt_cache_prefix_len > 0)

    async def _stream_openai(
        self,
        provider,
        api_key: str,
        model,
        messages: list[dict],
        system_prompt: str | None,
        temperature: float,
        max_tokens: int | None,
        tools: list[dict] | None,
        tool_choice: str | None,
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        """构建 OpenAI 兼容请求并解析 SSE 流，按 tool_call 索引聚合 function calling。

        OpenAI 兼容协议对稳定前缀自动缓存，无需主动干预。
        """
        payload = {
            "model": model.model_id,
            "messages": self._convert_messages(messages, system_prompt),
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens or model.max_output_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        url = provider.api_base_url.rstrip("/") + "/chat/completions"
        headers = build_auth_headers(provider, api_key)
        pending_tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments}
        async with self._client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    yield LlmStreamChunk(type="text_delta", content=delta["content"])
                # 推理模型思考流：openai_compat 推理模型（如 deepseek-r1）
                # 经 reasoning_content 增量下发
                if delta.get("reasoning_content"):
                    yield LlmStreamChunk(type="thinking_delta", content=delta["reasoning_content"])
                for tool_call in delta.get("tool_calls") or []:
                    index = tool_call.get("index", 0)
                    if index not in pending_tool_calls:
                        fn = tool_call.get("function") or {}
                        pending_tool_calls[index] = {
                            "id": tool_call.get("id", ""),
                            "name": fn.get("name", ""),
                            "arguments": "",
                        }
                        yield LlmStreamChunk(
                            type="tool_call_start",
                            tool_call_id=pending_tool_calls[index]["id"],
                            tool_call_name=pending_tool_calls[index]["name"],
                        )
                    tc = pending_tool_calls[index]
                    fn = tool_call.get("function") or {}
                    if fn.get("id") and not tc["id"]:
                        tc["id"] = fn["id"]
                    if fn.get("name") and not tc["name"]:
                        tc["name"] = fn["name"]
                    arguments = fn.get("arguments")
                    if arguments:
                        tc["arguments"] += arguments
                        yield LlmStreamChunk(type="tool_call_delta", content=arguments)
                if chunk.get("usage"):
                    yield LlmStreamChunk(type="done", usage=chunk["usage"])
            # 流式结束：对每个未完成的 tool_call 发 tool_call_complete
            for index in sorted(pending_tool_calls):
                tc = pending_tool_calls[index]
                yield LlmStreamChunk(
                    type="tool_call_complete",
                    content=tc["arguments"],
                    tool_call_id=tc["id"],
                    tool_call_name=tc["name"],
                )
            pending_tool_calls.clear()

    async def _stream_anthropic(
        self,
        provider,
        api_key: str,
        model,
        messages: list[dict],
        system_prompt: str | None,
        max_tokens: int | None,
        tools: list[dict] | None,
        tool_choice: str | None,
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        """构建 Anthropic 原生请求并解析 SSE 流，聚合 tool_use 内容块为三段式 tool_call 事件。

        模型启用 Prompt Caching 时，对稳定前缀（system + 最后一个工具定义）
        注入 cache_control，命中缓存按 cached_rate 计费。
        """
        payload = {
            "model": model.model_id,
            "messages": self._convert_messages_anthropic(messages),
            "stream": True,
            "max_tokens": max_tokens or model.max_output_tokens,
        }
        cache = self._should_cache(model)
        if system_prompt:
            if cache:
                # 稳定前缀：system 转内容块并标记 cache_control
                payload["system"] = [
                    {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                payload["system"] = system_prompt
        if tools is not None:
            anthropic_tools = self._convert_tools_anthropic(tools)
            if cache and anthropic_tools:
                # 工具定义为稳定前缀，对最后一个工具注入 cache_control
                anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = anthropic_tools
            anthropic_choice = self._convert_tool_choice_anthropic(tool_choice)
            if anthropic_choice is not None:
                payload["tool_choice"] = anthropic_choice
        url = provider.api_base_url.rstrip("/") + "/messages"
        headers = build_auth_headers(provider, api_key)
        headers.setdefault("anthropic-version", "2023-06-01")
        usage: dict = {}
        pending: dict[int, dict] = {}  # index -> {id, name, arguments}
        async with self._client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "message_start":
                    usage.update(event.get("message", {}).get("usage") or {})
                elif etype == "content_block_start":
                    cb = event.get("content_block") or {}
                    if cb.get("type") == "tool_use":
                        index = event.get("index", 0)
                        pending[index] = {
                            "id": cb.get("id", ""),
                            "name": cb.get("name", ""),
                            "arguments": "",
                        }
                        yield LlmStreamChunk(
                            type="tool_call_start",
                            tool_call_id=pending[index]["id"],
                            tool_call_name=pending[index]["name"],
                        )
                    elif cb.get("type") == "thinking":
                        # 推理模型思考流：initial thinking 文本一次下发，signature 丢弃
                        thinking = cb.get("thinking") or ""
                        if thinking:
                            yield LlmStreamChunk(type="thinking_delta", content=thinking)
                elif etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield LlmStreamChunk(type="text_delta", content=delta["text"])
                    elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                        yield LlmStreamChunk(type="thinking_delta", content=delta["thinking"])
                    elif delta.get("type") == "input_json_delta":
                        index = event.get("index", 0)
                        partial = delta.get("partial_json", "")
                        if index in pending:
                            pending[index]["arguments"] += partial
                            yield LlmStreamChunk(type="tool_call_delta", content=partial)
                elif etype == "content_block_stop":
                    index = event.get("index", 0)
                    if index in pending:
                        tc = pending.pop(index)
                        yield LlmStreamChunk(
                            type="tool_call_complete",
                            content=tc["arguments"],
                            tool_call_id=tc["id"],
                            tool_call_name=tc["name"],
                        )
                elif etype == "message_delta":
                    usage.update(event.get("usage") or {})
            yield LlmStreamChunk(type="done", usage=usage)

    async def _record_success(
        self,
        redis,
        provider_id: int,
        key_id: int,
        latency_ms: int,
        on_route_result: Callable[[dict], None] | None,
        model,
    ) -> None:
        """调用成功：Key 成功标记（含日计数/last_used）+ 供应商健康指标 + 归因透出"""
        # 无请求上下文（评测/A2A 临时会话等）时 contextvar 未设值，容忍为 None
        try:
            user_id = get_current_user_id()
        except LookupError:
            user_id = None
        await AiProviderKeyService.mark_call_success(redis, key_id, user_id)
        await provider_health_service.record_call(redis, provider_id, True, None, latency_ms)
        if on_route_result is not None:
            on_route_result(
                {
                    "model_id": model.model_id,
                    "model_pk": model.id,
                    "provider_id": provider_id,
                    "key_id": key_id,
                    "latency_ms": latency_ms,
                    "error_code": None,
                    "request_id": _get_trace_id(),
                }
            )

    async def _stream_with_key_retry(
        self,
        db,
        redis,
        provider,
        model,
        messages: list[dict],
        system_prompt: str | None,
        temperature: float,
        max_tokens: int | None,
        tools: list[dict] | None,
        tool_choice: str | None,
        on_route_result: Callable[[dict], None] | None,
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        """在一个候选路由内按 Key 优先级组逐 Key 尝试；全部 Key 失败抛 _RouteFailed。

        流式失败分两段：首字节前失败可切换下一 Key；流中断（已下发部分内容）
        标记 Key 失败后直接抛出业务异常，不重试整个请求。
        """
        started = time.perf_counter()
        keys = await AiProviderKeyService.list_usable_keys(db, redis, provider.id)
        if not keys:
            raise _RouteFailed("no_key", "该供应商无可用 API Key")

        last_error: tuple[str, str] = ("no_key", "该供应商无可用 API Key")
        for key in keys:
            key_id = key.id
            api_key = decrypt(key.key_cipher)
            first_chunk = True
            try:
                if provider.protocol_type == "anthropic":
                    stream = self._stream_anthropic(
                        provider,
                        api_key,
                        model,
                        messages,
                        system_prompt,
                        max_tokens,
                        tools,
                        tool_choice,
                    )
                else:  # openai_compat
                    stream = self._stream_openai(
                        provider,
                        api_key,
                        model,
                        messages,
                        system_prompt,
                        temperature,
                        max_tokens,
                        tools,
                        tool_choice,
                    )
                async for chunk in stream:
                    first_chunk = False
                    yield chunk
                # 流正常结束 → 记录成功并透出归因
                latency_ms = int((time.perf_counter() - started) * 1000)
                await self._record_success(
                    redis, provider.id, key_id, latency_ms, on_route_result, model
                )
                return
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                error_code, detail = _map_httpx_error(exc)
                latency_ms = int((time.perf_counter() - started) * 1000)
                is_local = provider.provider_code == "local"
                if not is_local:
                    await AiProviderKeyService.mark_call_failed(redis, key_id, error_code)
                    await provider_health_service.record_call(
                        redis, provider.id, False, error_code, latency_ms
                    )
                elif not first_chunk:
                    # 本地流中断：已下发部分内容，无法重放 → 直接抛出（不冷却占位 Key）
                    logger.error("本地 provider 流式响应中断: %s", detail)
                    raise BusinessException(
                        ResultCode.AI_LLM_CALL_FAILED, f"流式响应中断: {detail}"
                    ) from exc
                else:
                    # 内置本地 provider：就绪状态由 ensure_running 自管理，传输错误多为瞬时
                    # （模型加载/并发推理），且占位 Key 无鉴权语义；冷却它会让冷却期内的
                    # 所有本地推理请求因"无可用 API Key"失败。故本地不进入 Key 冷却，仅记录。
                    logger.warning("本地 provider 调用失败(%s): %s", error_code, detail)
                if not first_chunk and not is_local:
                    # 非本地流中断：已下发部分内容，无法重放 → 直接抛出，不切 Key / 不降级
                    logger.error("供应商 %s Key %s 流式响应中断: %s", provider.id, key_id, detail)
                    raise BusinessException(
                        ResultCode.AI_LLM_CALL_FAILED, f"流式响应中断: {detail}"
                    ) from exc
                logger.warning(
                    "供应商 %s Key %s 调用失败(%s)，切换下一 Key", provider.id, key_id, error_code
                )
                last_error = (error_code, detail)
                continue

        code, detail = last_error
        raise _RouteFailed(code, f"供应商 {provider.id} 全部 Key 不可用: {detail}")

    async def stream_chat(
        self,
        db,
        redis,
        model_id: str,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        on_route_result: Callable[[dict], None] | None = None,
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        """调用 LLM 并返回统一的流式响应。

        按「候选路由序列 + 逐候选尝试」调度：当前模型 → 同模型备用供应商 →
        降级链各级；候选路由内按 Key 优先级组逐 Key 重试。全部候选失败抛业务异常。

        tools/tool_choice 非 None 时启用 function calling；不传则与普通对话等价。

        on_route_result 可选：每次调用成功后回调一次，携带实际使用的
        model_id/provider_id/key_id/latency_ms/error_code/request_id（计费归因透出）。
        """
        # 能力要求：流式恒必；携带工具定义时要求工具调用能力
        required_caps = {"streaming"}
        if tools is not None:
            required_caps.add("tool_call")

        routes = await AiModelService.get_call_routes(db, model_id, required_caps)
        if not routes:
            raise BusinessException(ResultCode.AI_MODEL_NOT_AVAILABLE, "模型不可用或已禁用")

        last_error: _RouteFailed | None = None
        for route in routes:
            provider_id = route["provider_id"]
            if await provider_health_service.get_status(redis, provider_id) == "open":
                logger.warning("供应商 %s 熔断中，跳过该候选路由", provider_id)
                continue
            provider = await ai_provider_repository.get_by_id(db, provider_id)
            model = await ai_model_repository.get_by_id(db, route["model_pk"])
            if not provider or provider.status != 1 or not model:
                continue
            # 内置本地 provider：确保子进程服务就绪（含模型自动下载，可能较慢，
            # 线程化避免阻塞事件循环）
            if provider.provider_code == "local":
                await asyncio.to_thread(ensure_running)
            try:
                async for chunk in self._stream_with_key_retry(
                    db,
                    redis,
                    provider,
                    model,
                    messages,
                    system_prompt,
                    temperature,
                    max_tokens,
                    tools,
                    tool_choice,
                    on_route_result,
                ):
                    yield chunk
                return
            except _RouteFailed as exc:
                logger.warning(
                    "候选路由 %s(供应商 %s) 调用失败: %s", model.model_id, provider_id, exc.detail
                )
                last_error = exc
                continue

        code, detail = (
            (last_error.error_code, last_error.detail)
            if last_error
            else ("no_route", "无可用候选路由")
        )
        raise BusinessException(
            ResultCode.AI_LLM_CALL_FAILED, f"主模型和降级模型均不可用: {detail}"
        )

    async def count_tokens(self, text: str) -> int:
        """简单估算 token 数（字符数 / 4）"""
        return max(1, len(text) // 4)


llm_client = LlmClient()
