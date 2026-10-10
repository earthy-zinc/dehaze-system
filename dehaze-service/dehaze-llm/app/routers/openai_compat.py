"""OpenAI 兼容能力端点：对话 / 向量 / 重排 / 能力发现。

本模块只负责协议与响应形状，推理执行与模型装载在 app/inference.py；
鉴权由全局 BearerAuthMiddleware 承担，本模块不含免鉴权例外。
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import inference
from app.config import settings
from app.errors import ServiceError
from app.model_store import (
    CHAT_MODEL_ID,
    EMBEDDING_MODEL_ID,
    RERANK_MODEL_ID,
    is_chat_downloaded,
    is_embedding_downloaded,
    is_rerank_downloaded,
)

router = APIRouter()

_DEFAULT_MAX_OUTPUT_TOKENS = 1024
# 向量模型上下文（与编排侧播种的 qwen3-embedding-0.6b max_context_tokens 一致）
_EMBEDDING_CTX_SIZE = 4096
_EMBEDDING_DIMENSION = 1024


# ===== 对话 =====


class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = CHAT_MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = _DEFAULT_MAX_OUTPUT_TOKENS
    # OpenAI 兼容字段（本地小模型不解析工具协议，仅保证接口形状兼容不报错）
    tools: list[dict] | None = None
    tool_choice: Any = None


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    llm = await inference.load("chat")
    prompt = inference.build_prompt([(m.role, m.content or "") for m in req.messages])
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    temperature = req.temperature if req.temperature is not None else 0.7
    max_tokens = req.max_tokens or _DEFAULT_MAX_OUTPUT_TOKENS

    if not req.stream:
        content, usage = await inference.chat_completion(
            llm, prompt, max_tokens=max_tokens, temperature=temperature
        )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    async def event_stream() -> AsyncGenerator[str, None]:
        usage: dict = {}
        async for item in inference.stream_chat(
            llm, prompt, max_tokens=max_tokens, temperature=temperature
        ):
            if isinstance(item, inference.StreamEnd):
                usage = item.usage
                break
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        tail = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
        # OpenAI 协议：usage 随独立 chunk（choices=[]）在 [DONE] 前下发，供客户端计量 token
        if usage:
            usage_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ===== 向量 =====


class EmbeddingRequest(BaseModel):
    model: str = EMBEDDING_MODEL_ID
    # OpenAI 兼容：单字符串或字符串数组，统一走 Qwen3-Embedding-0.6B（1024 维）
    input: str | list[str]


@router.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    llm = await inference.load("embedding")
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    vectors, prompt_tokens = await inference.embed(llm, texts)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vector}
            for i, vector in enumerate(vectors)
        ],
        "model": req.model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


# ===== 重排 =====


class RerankRequest(BaseModel):
    """Jina/Cohere 兼容 rerank 请求（对齐 rerank_service 的请求字段）。"""

    model: str = RERANK_MODEL_ID
    query: str
    documents: list[str]
    top_n: int | None = None


@router.post("/v1/rerank")
async def rerank(req: RerankRequest):
    """OpenAI/Jina 兼容 rerank 端点，供编排侧 rerank_service 按 provider 的
    api_base_url（.../v1）+ /rerank 派生直连；响应 results 按分数降序。
    """
    if not req.documents:
        return {"model": req.model, "results": []}
    if not is_rerank_downloaded():
        # 可选模型不随启动预热：首次请求触发后台下载并立返 503，客户端稍后重试即可
        inference.trigger_rerank_download()
        raise ServiceError(
            503,
            "MODEL_NOT_READY",
            "重排模型未下载，已触发后台自动下载（约 610MB），请稍后重试",
        )
    llm = await inference.load("rerank")
    scores = await inference.rerank_scores(llm, req.query, req.documents)
    results = [{"index": i, "relevance_score": score} for i, score in enumerate(scores)]
    results.sort(key=lambda r: r["relevance_score"], reverse=True)
    if req.top_n:
        results = results[: req.top_n]
    return {"model": req.model, "results": results}


# ===== 能力发现 =====


def _model_card(model_id: str, model_type: str, downloaded: bool, reason: str) -> dict:
    return {
        "id": model_id,
        "type": model_type,
        "available": downloaded,
        "unavailable_reason": None if downloaded else reason,
    }


def _model_cards() -> list[dict]:
    """本服务可服务的全部模型及其运行时能力（只读，不触发下载/加载）。"""
    chat_ready = is_chat_downloaded()
    embed_ready = is_embedding_downloaded()
    rerank_ready = is_rerank_downloaded()
    concurrency = {"max_concurrency": 1}  # 对话/向量/重排共用一把推理锁，串行执行
    return [
        {
            **_model_card(CHAT_MODEL_ID, "chat", chat_ready, "模型文件未下载（服务预热中）"),
            "capabilities": {"streaming": True},
            "limits": {
                "context_tokens": settings.CTX_SIZE,
                "max_output_tokens": _DEFAULT_MAX_OUTPUT_TOKENS,
                **concurrency,
            },
        },
        {
            **_model_card(
                EMBEDDING_MODEL_ID, "embedding", embed_ready, "模型文件未下载（服务预热中）"
            ),
            "capabilities": {"dimensions": _EMBEDDING_DIMENSION},
            "limits": {"context_tokens": _EMBEDDING_CTX_SIZE, **concurrency},
        },
        {
            **_model_card(
                RERANK_MODEL_ID,
                "rerank",
                rerank_ready,
                "模型文件未下载，首次重排请求将触发后台下载",
            ),
            "capabilities": {"top_n": True},
            "limits": {"context_tokens": settings.CTX_SIZE, **concurrency},
        },
    ]


@router.get("/v1/models")
async def list_models() -> dict:
    return {"object": "list", "data": _model_cards()}


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> dict:
    for card in _model_cards():
        if card["id"] == model_id:
            return card
    raise ServiceError(404, "MODEL_NOT_FOUND", f"模型 {model_id} 不存在")
