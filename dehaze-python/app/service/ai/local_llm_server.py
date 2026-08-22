"""内置本地轻量 LLM 服务（OpenAI 兼容，独立子进程运行）

设计要点：
- 以子进程运行：CPU 密集的 llama.cpp 推理会阻塞 FastAPI 事件循环，必须进程隔离
- OpenAI 兼容 /v1/chat/completions（含流式 SSE）：作为普通 provider 被 LlmClient 调用，
  Key 轮换/健康度/降级链/调用审计全链路保持真实
- Qwen3 关闭思考模式：system 尾部注入 /no_think 软指令，输出层再剥离 <think> 残留，
  保证消息内容干净
- 模型懒加载单例：进程启动后首次请求时加载 GGUF
- GPU 优先推理：llama-cpp-python 为 CUDA 构建时全量卸载至 GPU（自动检测，纯 CPU
  构建回退 CPU）。GPU 推理快且不占满 CPU 核心，/health 等管理面请求不受争用拖慢
- 推理串行化：本地单模型推理，llama.cpp 并发调用不稳定，对话与嵌入共用
  asyncio.Lock，同一时刻仅一个推理执行，其余排队（排队等待可被取消）
- 断连即停：流式请求由工作线程独占推理锁并逐 token 投递，客户端断连后置停止标志，
  工作线程在当前 token 完成后退出并释放锁——已断连的请求不再空转生成剩余 token，
  排队中的请求也不会被僵尸请求长时间饿死（锁释放滞后最多约 0.5 秒）
- 配置统一走 app.config（pydantic-settings 天然支持环境变量/.env 覆盖）

启动方式：
    python -m app.service.ai.local_llm_server  （由 local_llm_manager 在需要时自动拉起，
    也可独立部署为共享推理服务）
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.service.ai.local_llm_model import embedding_model_path, model_path

logger = logging.getLogger(__name__)

app = FastAPI(title="Dehaze Local LLM", docs_url=None, redoc_url=None)

_llm: Any = None
_llm_lock = threading.Lock()


def _n_gpu_layers() -> int:
    """GPU 卸载层数：自动检测（CUDA 构建时全量卸载 -1，纯 CPU 构建为 0），
    LOCAL_LLM_NGPU_LAYERS 显式配置时优先。"""
    if settings.LOCAL_LLM_NGPU_LAYERS is not None:
        return settings.LOCAL_LLM_NGPU_LAYERS
    try:
        from llama_cpp import llama_cpp

        return -1 if llama_cpp.llama_supports_gpu_offload() else 0
    except Exception:  # noqa: BLE001 检测失败按纯 CPU 处理
        return 0

# 推理串行化锁：本地单模型 CPU 推理，llama.cpp create_completion 并发调用不稳定，
# 对话与嵌入共用同一把锁，保证每次只有一个推理在执行，其余排队。
# 用 asyncio.Lock 使排队等待可被取消：客户端断连时在 await 处直接放弃排队。
_infer_lock = asyncio.Lock()

_embed_llm: Any = None
_embed_lock = threading.Lock()

# 流式生成工作线程登记表：持有强引用防止 fire-and-forget 任务被 GC 中途回收
_active_workers: set[asyncio.Task] = set()


def _get_llm() -> Any:
    """懒加载 GGUF 模型单例（约 378MB，Q4_K_M 量化）。

    用锁保证并发首次请求（health 探测 / chat 调用同时到达）只有一个线程加载模型，
    其余线程等待，避免重复实例化 Llama 引发崩溃或连接异常。
    """
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                from llama_cpp import Llama

                path = model_path()
                if not os.path.exists(path):
                    raise RuntimeError(
                        f"本地模型文件不存在: {path}，请检查 local_llm_manager 是否正确拉起本进程"
                    )
                n_gpu = _n_gpu_layers()
                logger.info("加载本地模型: %s（GPU 卸载层数 %s）", path, n_gpu)
                started = time.perf_counter()
                _llm = Llama(
                    model_path=path,
                    n_ctx=settings.LOCAL_LLM_CTX_SIZE,
                    n_threads=settings.LOCAL_LLM_THREADS or None,
                    n_gpu_layers=n_gpu,
                    verbose=False,
                )
                logger.info("模型加载完成，耗时 %.1fs", time.perf_counter() - started)
    return _llm


def _get_embed_llm() -> Any:
    """懒加载 GGUF 向量模型单例（Qwen3-Embedding-0.6B，1024 维），与对话模型独立。"""
    global _embed_llm
    if _embed_llm is None:
        with _embed_lock:
            if _embed_llm is None:
                from llama_cpp import Llama

                path = embedding_model_path()
                if not os.path.exists(path):
                    raise RuntimeError(
                        f"本地向量模型文件不存在: {path}，请检查 local_llm_model.ensure_embedding_model"
                    )
                n_gpu = _n_gpu_layers()
                logger.info("加载本地向量模型: %s（GPU 卸载层数 %s）", path, n_gpu)
                started = time.perf_counter()
                _embed_llm = Llama(
                    model_path=path,
                    n_threads=settings.LOCAL_LLM_THREADS or None,
                    n_gpu_layers=n_gpu,
                    embedding=True,
                    verbose=False,
                )
                logger.info("向量模型加载完成，耗时 %.1fs", time.perf_counter() - started)
    return _embed_llm


class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-0.6b"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = 1024
    # OpenAI 兼容字段（本地小模型不解析工具协议，仅保证接口形状兼容不报错）
    tools: list[dict] | None = None
    tool_choice: Any = None


def _build_prompt(messages: list[ChatMessage]) -> str:
    """ChatML 格式；首条 system 追加 /no_think 关闭 Qwen3 思考模式"""
    tmpl = "<|im_start|>{role}\n{content}<|im_end|>\n"
    parts = []
    for i, m in enumerate(messages):
        content = m.content or ""
        if m.role == "system" and i == 0:
            content = f"{content} /no_think" if content else "/no_think"
        parts.append(tmpl.format(role=m.role, content=content))
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


_STOP_TOKENS = ("<|im_end|>", "<|im_start|>")


def _clean_content(text: str) -> str:
    """剥离 Qwen3 可能漏出的思考块与模板残留"""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    for tok in _STOP_TOKENS:
        text = text.split(tok)[0]
    return text.lstrip()


@app.get("/health")
def health() -> dict:
    """就绪探测：未加载时触发懒加载，使 manager 的轮询在模型真正就绪后才放行。

    首次 /health 会阻塞直至模型加载完成（约 5-15s），返回 loaded=true；
    这样 local_llm_manager.ensure_running 能等到模型就绪，而非在进程刚起、模型未载时
    提前返回，从而避免首个推理请求因模型未就绪而连接失败。
    """
    loaded = _llm is not None
    if not loaded:
        try:
            _get_llm()
            loaded = True
        except Exception as exc:  # noqa: BLE001 探测路径不抛，交由调用方按未就绪处理
            logger.warning("健康探测触发模型加载失败: %s", exc)
            loaded = False
    return {"status": "ok", "model": "qwen3-0.6b", "loaded": loaded}


class EmbeddingRequest(BaseModel):
    model: str = "qwen3-embedding-0.6b"
    # OpenAI 兼容：单字符串或字符串数组；统一走本地 Qwen3-Embedding-0.6B（1024 维）
    input: str | list[str]


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    llm = await asyncio.to_thread(_get_embed_llm)
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    data = []
    prompt_tokens = 0
    for i, text in enumerate(texts):
        # 逐条持锁：长文档批量嵌入不长时间独占推理锁，对话请求可公平穿插
        async with _infer_lock:
            result = await asyncio.to_thread(llm.create_embedding, text)
        data.append({"object": "embedding", "index": i, "embedding": result["data"][0]["embedding"]})
        prompt_tokens += result.get("usage", {}).get("prompt_tokens", 0)
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


# 流式生成结束标记（区别于任何正常 token 文本与异常对象）
_STREAM_DONE = object()


def _offer(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
    item: Any,
    stop: threading.Event,
) -> bool:
    """工作线程向异步队列投递元素；消费者已退出（stop 置位）时放弃。

    每次最多等 0.5s 后复查 stop，避免消费者断开后投递协程在满队列上永久挂起。
    """
    while True:
        fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        try:
            fut.result(timeout=0.5)
            return True
        except TimeoutError:
            if stop.is_set():
                fut.cancel()
                return False


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    llm = await asyncio.to_thread(_get_llm)
    prompt = _build_prompt(req.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    temperature = req.temperature if req.temperature is not None else 0.7
    max_tokens = req.max_tokens or 1024

    if not req.stream:
        async with _infer_lock:
            result = await asyncio.to_thread(
                llm.create_completion,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=list(_STOP_TOKENS),
            )
        content = _clean_content(result["choices"][0]["text"])
        usage = result.get("usage", {})
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    async def event_stream():
        # 排队阶段即可取消：客户端断连时在锁等待处直接结束，不为死连接排队
        await _infer_lock.acquire()
        stop = threading.Event()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        loop = asyncio.get_running_loop()

        def generate():
            """工作线程：独占推理锁逐 token 生成并投递；stop 置位后当前 token 完成即退出。"""
            error: Exception | None = None
            try:
                for chunk in llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=list(_STOP_TOKENS),
                    stream=True,
                ):
                    if stop.is_set():
                        return
                    delta = _clean_content(chunk["choices"][0].get("text", ""))
                    if delta and not _offer(loop, queue, delta, stop):
                        return
            except Exception as exc:
                # 生成异常经队列回传消费方中断 SSE，让 provider 健康度统计感知失败
                error = exc
            finally:
                # 锁由生成线程释放：新请求拿到锁时本请求生成必然已完全停止，
                # 不存在两个线程同时进入 llama.cpp（非线程安全）的窗口
                _infer_lock.release()
                _offer(loop, queue, error if error is not None else _STREAM_DONE, stop)

        worker = asyncio.create_task(asyncio.to_thread(generate))
        _active_workers.add(worker)
        worker.add_done_callback(_active_workers.discard)
        try:
            while True:
                item = await queue.get()
                if item is _STREAM_DONE:
                    break
                if isinstance(item, Exception):
                    raise item
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
            yield "data: [DONE]\n\n"
        finally:
            # 客户端断连（任务取消）或正常结束时通知生成线程停止
            stop.set()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(
        app,
        host=settings.LOCAL_LLM_HOST,
        port=settings.LOCAL_LLM_PORT,
        log_level="warning",
    )
