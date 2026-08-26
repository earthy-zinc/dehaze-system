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
import logging
import time
from collections.abc import AsyncGenerator, Callable

import httpx
from redis.asyncio import Redis

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import _get_trace_id
from app.dependencies.redis import get_redis_client
from app.infrastructure.crypto.aes_cipher import decrypt
from app.infrastructure.llm.local.local_llm_manager import ensure_running
from app.infrastructure.llm.common import LlmStreamChunk, _map_httpx_error
from app.infrastructure.llm.client.model_client import create_chat_client
from app.infrastructure.provider.model_registry import model_registry
from app.infrastructure.provider.provider_health_service import provider_health_service
from app.infrastructure.provider.provider_key_selector import provider_key_selector
from app.models.base import get_current_user_id
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_provider_repository import ai_provider_repository

logger = logging.getLogger(__name__)


class _RouteFailed(Exception):
    """某候选路由（供应商）全部 Key 调用失败，用于切换下一候选路由"""

    def __init__(self, error_code: str, detail: str) -> None:
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail


class LlmClient:
    """LLM 客户端（单例）"""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(settings.AI_MESSAGE_STREAM_TIMEOUT))
        self._redis: Redis | None = None

    async def _get_redis(self) -> Redis:
        """惰性获取全局 Redis 客户端（get_redis_client 为单例，仅首次 await 建连）"""
        if self._redis is None:
            self._redis = await get_redis_client()
        return self._redis

    async def _record_success(
        self,
        provider_id: int,
        key_id: int,
        latency_ms: int,
        on_route_result: Callable[[dict], None] | None,
        model,
    ) -> None:
        """调用成功：Key 成功标记（含日计数/last_used）+ 供应商健康指标 + 归因透出"""
        redis = await self._get_redis()
        # 无请求上下文（评测/A2A 临时会话等）时 contextvar 未设值，容忍为 None
        try:
            user_id = get_current_user_id()
        except LookupError:
            user_id = None
        await provider_key_selector.mark_call_success(redis, key_id, user_id)
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
        redis = await self._get_redis()
        started = time.perf_counter()
        keys = await provider_key_selector.list_usable_keys(db, redis, provider.id)
        if not keys:
            raise _RouteFailed("no_key", "该供应商无可用 API Key")

        last_error: tuple[str, str] = ("no_key", "该供应商无可用 API Key")
        chat_client = create_chat_client(provider.protocol_type, self._client)
        for key in keys:
            key_id = key.id
            api_key = decrypt(key.key_cipher)
            first_chunk = True
            try:
                stream = chat_client.stream_chat(
                    provider,
                    api_key,
                    model,
                    messages,
                    system_prompt,
                    max_tokens,
                    tools,
                    tool_choice,
                    temperature,
                )
                async for chunk in stream:
                    first_chunk = False
                    yield chunk
                # 流正常结束 → 记录成功并透出归因
                latency_ms = int((time.perf_counter() - started) * 1000)
                await self._record_success(provider.id, key_id, latency_ms, on_route_result, model)
                return
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                error_code, detail = _map_httpx_error(exc)
                latency_ms = int((time.perf_counter() - started) * 1000)
                is_local = provider.provider_code == "local"
                if not is_local:
                    await provider_key_selector.mark_call_failed(redis, key_id, error_code)
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

        redis 为实例级基础设施依赖：惰性获取全局单例后由类内私有方法
        共享同一引用，不向下透传；provider_health_service / provider_key_selector
        等原子逻辑仍以显式参数接收。
        """
        redis = await self._get_redis()

        # 能力要求：流式恒必；携带工具定义时要求工具调用能力
        required_caps = {"streaming"}
        if tools is not None:
            required_caps.add("tool_call")

        routes = await model_registry.get_call_routes(db, model_id, required_caps)
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
