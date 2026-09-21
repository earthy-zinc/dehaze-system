"""AI 模型可用性测试服务

按模型类型对单个模型发最小真实推理请求，验证模型在供应商侧真实开通可用
（供应商连通性测试只验证供应商可访问与凭据有效，验证不了具体 model_id 开通）。

- chat：openai_compat 走 /chat/completions（max_tokens=1），anthropic 走 /v1/messages
- embedding：/embeddings（cohere 特判 /v1/embed），单词向量化
- rerank：/rerank 单文档重排（OpenAI 兼容，同 kb rerank_service 口径）
- local 供应商（内置本地模型）复用同一探测逻辑：local 服务的 base_url 已配置为
  OpenAI 兼容端点，且无 rerank 模型注册

测试为管理员手动触发，不落业务计费（与供应商连通性测试同口径），外部 API 侧
真实成本为最小请求的极少 token。结果落 sys_ai_model.last_test_* 三列，模型列表
直接展示，供管理员"上架前验证"与故障定位（需求规格 §2.8 上架前建议连通性验证）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta

import httpx
from sqlalchemy import bindparam, text

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.llm.common import build_auth_headers
from app.infrastructure.llm.local.local_llm_manager import ensure_running
from app.infrastructure.provider.provider_key_selector import provider_key_selector
from app.models.entity.sys_ai_model import SysAiModel
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_provider_repository import ai_provider_repository
from app.service.ai_model_service import _clear_model_cache

logger = logging.getLogger(__name__)

_TEST_TIMEOUT = 30.0
_ERROR_MAX_LEN = 500


def _probe_request(
    protocol_type: str, model_type: str, base_url: str, model: str
) -> tuple[str, dict]:
    """按协议与模型类型组装最小探测 (url, payload)"""
    base = base_url.rstrip("/")
    if model_type == "embedding":
        if base.endswith("/v1/embed"):
            return base, {"model": model, "input": "hi", "input_type": "search_query"}
        return f"{base}/embeddings", {"model": model, "input": "hi"}
    if model_type == "rerank":
        url = base if base.endswith("/rerank") else f"{base}/rerank"
        return url, {"model": model, "query": "hi", "documents": ["hello"], "top_n": 1}
    # chat
    if protocol_type == "anthropic":
        return (
            f"{base}/v1/messages",
            {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
        )
    return (
        f"{base}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": False,
        },
    )


def _brief_error(prefix: str, resp: httpx.Response) -> str:
    """从错误响应体提取简短信息（截断防溢出 varchar(500)）"""
    try:
        body = resp.json()
        detail = body.get("error") or body.get("message") or ""
        if isinstance(detail, dict):
            detail = detail.get("message", str(detail))
        return f"{prefix}: {str(detail)[:200]}" if detail else prefix
    except (ValueError, AttributeError):
        # 响应体非合法 JSON 或非对象（网关 HTML/空体/数组）：仅回退到状态码前缀，
        # 记日志保留线索，避免"为何拿不到供应商错误详情"无迹可查
        logger.debug(
            "模型测试错误响应体无法解析为 JSON: status=%s", getattr(resp, "status_code", None)
        )
        return prefix


async def test_model(db, redis, model_pk: int) -> dict:
    """对单个模型执行可用性测试，结果落库并返回 {success, latencyMs, error}。

    测试失败不抛异常（结果本身就是"不可用"），模型不存在/供应商不可用才抛业务异常。
    """
    model = await ai_model_repository.get_by_id(db, model_pk)
    if not model:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "模型不存在")
    provider = await ai_provider_repository.get_by_id(db, model.provider_id)
    if not provider or provider.status != 1:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "所属供应商不存在或已禁用")
    if provider.provider_code == "local" and model.model_type != "rerank":
        # 本地推理服务与主进程分离，调用前确保已拉起（PDEATHSIG 回收后需重新拉起）
        await asyncio.to_thread(ensure_running)
    api_key = await provider_key_selector.select_key(db, redis, provider.id)
    if not api_key:
        raise BusinessException(
            ResultCode.OPERATION_NOT_ALLOW, "所属供应商没有可用的启用 API Key，无法测试"
        )

    url, payload = _probe_request(
        provider.protocol_type, model.model_type, provider.api_base_url, model.model_id
    )
    headers = build_auth_headers(provider, api_key)

    result = {"success": False, "latencyMs": None, "error": None}
    try:
        async with httpx.AsyncClient(timeout=_TEST_TIMEOUT) as client:
            start = time.monotonic()
            resp = await client.post(url, headers=headers, json=payload)
            result["latencyMs"] = int((time.monotonic() - start) * 1000)
            if resp.status_code < 400:
                result["success"] = True
            else:
                result["error"] = _brief_error(f"HTTP {resp.status_code}", resp)
    except httpx.TimeoutException:
        result["error"] = f"连接超时（{int(_TEST_TIMEOUT)}s）"
    except httpx.HTTPError as exc:
        result["error"] = f"请求失败: {type(exc).__name__}"
    except Exception as exc:
        result["error"] = f"连接失败: {exc}"
        logger.warning("模型可用性测试异常 model=%s: %s", model.model_id, exc)

    if result["success"]:
        error_text = None
    else:
        error_text = (result["error"] or "未知错误")[:_ERROR_MAX_LEN]
        if result["latencyMs"] is not None:
            error_text = f"{error_text}（延迟 {result['latencyMs']}ms）"

    model.last_test_status = 1 if result["success"] else 2
    model.last_test_at = datetime.now()
    model.last_test_error = error_text
    await db.flush()
    # 测试结果落在启用模型缓存快照里，刷新缓存保证消费方列表可见
    await _clear_model_cache(redis)
    return result


async def test_model_by_model_id(db, redis, model_id: str) -> dict:
    """按业务 model_id 定位模型并测试（router 层复用现有 {model_id} 路径约定）"""
    model = await ai_model_repository.get_by_model_id(db, model_id)
    if not model:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "模型不存在")
    return await test_model(db, redis, model.id)


async def get_usage_stats_24h(db, models: list[SysAiModel]) -> dict[int, dict]:
    """批量查询模型近 24h 真实调用统计（模型列表 B 侧展示）。

    chat 模型按 sys_ai_llm_call 聚合（含失败/超时，成功率真实反映）；
    embedding/rerank 按 sys_ai_billing 计费流水聚合（成功调用才落账，恒为可用佐证）。
    返回 {model_pk: {calls_24h, success_rate_24h, last_call_at}}，无调用的模型不在结果中。
    """
    since = datetime.now() - timedelta(hours=24)
    stats: dict[int, dict] = {}

    chat_models = [m for m in models if m.model_type == "chat"]
    if chat_models:
        rows = (
            (
                await db.execute(
                    text(
                        "SELECT model, COUNT(*) AS total, "
                        "SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) AS ok, "
                        "MAX(create_time) AS last_at "
                        "FROM sys_ai_llm_call WHERE create_time >= :since AND model IN :models "
                        "GROUP BY model"
                    ).bindparams(bindparam("models", expanding=True)),
                    {"since": since, "models": [m.model_id for m in chat_models]},
                )
            )
            .mappings()
            .all()
        )
        pk_by_model_id = {m.model_id: m.id for m in chat_models}
        for row in rows:
            pk = pk_by_model_id.get(row["model"])
            if pk is None:
                continue
            total = int(row["total"] or 0)
            ok = int(row["ok"] or 0)
            stats[pk] = {
                "calls_24h": total,
                "success_rate_24h": round(ok * 100 / total) if total else None,
                "last_call_at": row["last_at"],
            }

    kb_models = [m for m in models if m.model_type in ("embedding", "rerank")]
    if kb_models:
        rows = (
            (
                await db.execute(
                    text(
                        "SELECT model, COUNT(*) AS total, MAX(create_time) AS last_at "
                        "FROM sys_ai_billing WHERE create_time >= :since "
                        "AND bill_type IN ('embedding', 'rerank') AND model IN :models "
                        "GROUP BY model"
                    ).bindparams(bindparam("models", expanding=True)),
                    {"since": since, "models": [m.model_id for m in kb_models]},
                )
            )
            .mappings()
            .all()
        )
        pk_by_model_id = {m.model_id: m.id for m in kb_models}
        for row in rows:
            pk = pk_by_model_id.get(row["model"])
            if pk is None:
                continue
            existing = stats.get(
                pk, {"calls_24h": 0, "success_rate_24h": None, "last_call_at": None}
            )
            existing["calls_24h"] += int(row["total"] or 0)
            if row["last_at"] and (
                existing["last_call_at"] is None or row["last_at"] > existing["last_call_at"]
            ):
                existing["last_call_at"] = row["last_at"]
            existing["success_rate_24h"] = 100 if existing["calls_24h"] else None
            stats[pk] = existing

    return stats
