"""推理核心：模型装载、推理锁、流式生成与断连停止（端点定义见 app/routers/openai_compat.py）。

设计要点：
- 模型懒加载单例：三个 GGUF 各自独立实例（llama.cpp 的 Llama 非线程安全），
  进程启动后异步预热对话与向量模型，预热完成前 /ready 返回 503
- 推理串行化：对话/向量/重排共用一把 asyncio.Lock，同一时刻仅一个推理执行，
  其余排队；用 asyncio.Lock 使排队等待可被客户端断连取消
- 断连即停：流式请求由工作线程独占推理锁并逐 token 投递，客户端断连后置停止标志，
  工作线程在当前 token 完成后退出并释放锁——已断连的请求不再空转生成剩余 token，
  排队中的请求也不会被僵尸请求长时间饿死（锁释放滞后最多约 0.5 秒）
- GPU 优先推理：llama-cpp-python 为 CUDA 构建时全量卸载至 GPU（自动检测，纯 CPU
  构建回退 CPU），GPU 推理不占满 CPU 核心
- Qwen3 关闭思考模式：system 尾部注入 /no_think 软指令，输出层再剥离模板残留
"""

import asyncio
import logging
import math
import threading
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from pathlib import Path
from typing import Any

from app.config import settings
from app.errors import ServiceError
from app.metrics import observe_inference
from app.model_store import (
    chat_model_path,
    embedding_model_path,
    ensure_chat_model,
    ensure_embedding_model,
    ensure_rerank_model,
    rerank_model_path,
)

logger = logging.getLogger(__name__)

# 推理串行化锁：本地单模型推理，llama.cpp 并发调用不稳定，对话/向量/重排共用一把锁
_infer_lock = asyncio.Lock()

# 流式生成工作线程登记表：持有强引用防止 fire-and-forget 任务被 GC 中途回收
_active_workers: set[asyncio.Task] = set()

_PREWARM_RETRY_SECONDS = 30


def _n_gpu_layers() -> int:
    """GPU 卸载层数：NGPU_LAYERS 显式配置时优先；否则 CUDA 构建全量卸载 -1，纯 CPU 构建 0。"""
    if settings.NGPU_LAYERS is not None:
        return settings.NGPU_LAYERS
    try:
        from llama_cpp import llama_cpp

        return -1 if llama_cpp.llama_supports_gpu_offload() else 0
    except (ImportError, AttributeError, OSError) as exc:
        # 只吞探测自身可能抛出的异常：延迟导入缺依赖 ImportError、缺符号 AttributeError、
        # CUDA 动态库加载失败 OSError。静默吞掉会让 CUDA 构建因环境问题退化为纯 CPU
        # （慢几十倍）而毫无痕迹，故留痕；未预期异常任其冒泡暴露
        logger.warning("GPU 卸载能力探测失败，按纯 CPU 运行: %s", exc, exc_info=True)
        return 0


class _ModelSlot:
    """单模型懒加载槽：进程内单例 + 加载锁。

    同一进程多线程并发首次请求（预热与推理请求同时到达）时只有一个线程实例化 Llama，
    其余等待，避免重复加载引发内存翻倍或崩溃。
    """

    def __init__(
        self,
        name: str,
        path_getter: Callable[[], str],
        ensure: Callable[[], str],
        **load_kwargs: Any,
    ) -> None:
        self.name = name
        self._path_getter = path_getter
        self._ensure = ensure
        self._load_kwargs = load_kwargs
        self._llm: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> Any:
        if self._llm is not None:
            return self._llm
        with self._lock:
            if self._llm is not None:
                return self._llm
            from llama_cpp import Llama

            path = self._ensure()  # 文件缺失时下载（镜像回退/断点续传/跨进程锁）
            if not Path(path).exists():
                raise RuntimeError(f"模型文件不存在: {path}，请检查服务侧的模型路径配置与下载日志")
            logger.info("加载模型 %s: %s（GPU 卸载层数 %s）", self.name, path, _n_gpu_layers())
            started = time.perf_counter()
            self._llm = Llama(
                model_path=path,
                n_threads=settings.THREADS or None,
                n_gpu_layers=_n_gpu_layers(),
                verbose=False,
                **self._load_kwargs,
            )
            logger.info("模型 %s 加载完成，耗时 %.1fs", self.name, time.perf_counter() - started)
            return self._llm


_chat = _ModelSlot("chat", chat_model_path, ensure_chat_model, n_ctx=settings.CTX_SIZE)
_embed = _ModelSlot("embedding", embedding_model_path, ensure_embedding_model, embedding=True)
# 重排 n_ctx 与对话一致：rerank 输入 = query + 文档节全文拼接，节内容可达数千 token，
# 小上下文会频繁截断文档导致相关信号丢失
_rerank = _ModelSlot("rerank", rerank_model_path, ensure_rerank_model, n_ctx=settings.CTX_SIZE)

_SLOTS: dict[str, _ModelSlot] = {"chat": _chat, "embedding": _embed, "rerank": _rerank}


def model_status() -> dict[str, bool]:
    """各模型当前装载状态（只读，不触发加载/下载）"""
    return {"chat": _chat.loaded, "embedding": _embed.loaded, "rerank": _rerank.loaded}


async def prewarm() -> None:
    """异步预热对话与向量模型（重排属可选能力不预热）。

    不阻塞进程启动；下载或加载失败每 30s 重试，避免瞬时网络故障导致服务永久不就绪。
    """
    for slot in (_chat, _embed):
        while not slot.loaded:
            try:
                await asyncio.to_thread(slot.load)
            except Exception as exc:
                logger.error(
                    "模型 %s 预热失败，%ss 后重试: %s", slot.name, _PREWARM_RETRY_SECONDS, exc
                )
                await asyncio.sleep(_PREWARM_RETRY_SECONDS)
    logger.info("对话与向量模型预热完成")


async def load(model_type: str) -> Any:
    """装载模型（阻塞部分走线程池）；失败按"模型不可用"返回 503（编排侧有限重试）。

    必须在响应头下发前调用：装载失败要以 HTTP 503 返回，而非流式响应中途断流。
    """
    try:
        return await asyncio.to_thread(_SLOTS[model_type].load)
    except Exception as exc:
        logger.error("模型 %s 装载失败: %s", model_type, exc, exc_info=True)
        raise ServiceError(503, "MODEL_NOT_READY", f"模型 {model_type} 不可用: {exc}") from exc


# ===== 对话 =====

_STOP_TOKENS = ("<|im_end|>", "<|im_start|>")


def build_prompt(messages: Sequence[tuple[str, str]]) -> str:
    """ChatML 格式；首条 system 追加 /no_think 关闭 Qwen3 思考模式"""
    tmpl = "<|im_start|>{role}\n{content}<|im_end|>\n"
    parts = []
    for i, (role, content) in enumerate(messages):
        if role == "system" and i == 0:
            content = f"{content} /no_think" if content else "/no_think"
        parts.append(tmpl.format(role=role, content=content))
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _clean_content(text: str) -> str:
    """剥离 Qwen3 可能漏出的模板残留。

    思考块 <think>...</think> 语义由消费端（编排侧 openai_compat 的 _ThinkSplitter）
    跨流式分块识别并转为思考流（前端思考区渲染），本函数**不处理**思考标签：
    流式逐 chunk 调用时标签可能被拆到多个 chunk、思考内容的 chunk 不含闭合标签，
    若在此剥离会把思考内容误当正文输出（历史 bug：消息正文残留思考）。
    """
    for tok in _STOP_TOKENS:
        text = text.split(tok)[0]
    return text.lstrip()


class StreamEnd:
    """流式生成结束标记（区别于任何正常 token 文本与异常对象），携带最终 usage"""

    __slots__ = ("usage",)

    def __init__(self, usage: dict):
        self.usage = usage


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


async def chat_completion(
    llm: Any, prompt: str, *, max_tokens: int, temperature: float
) -> tuple[str, dict]:
    """非流式对话：返回 (正文, usage)"""
    with observe_inference("chat"):
        async with _infer_lock:
            result = await asyncio.to_thread(
                llm.create_completion,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=list(_STOP_TOKENS),
            )
    return _clean_content(result["choices"][0]["text"]), result.get("usage", {})


async def stream_chat(
    llm: Any, prompt: str, *, max_tokens: int, temperature: float
) -> AsyncGenerator[str | StreamEnd, None]:
    """流式对话：逐 token 产出正文增量，末尾产出 StreamEnd（携带 usage）。

    模型装载由调用方在响应头下发前完成；推理锁的获取在生成开始时（排队阶段可被客户端
    断连取消），释放在生成线程退出时——新请求拿到锁时本请求生成必然已完全停止。
    """
    await _infer_lock.acquire()
    stop = threading.Event()
    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    loop = asyncio.get_running_loop()
    # llama.cpp 流式 chunk 不携带 usage：prompt 侧按 tokenize 计数（与推理输入一致），
    # completion 侧流式每 chunk 恰为一个 token，按 chunk 数累计
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))

    def generate() -> None:
        """工作线程：独占推理锁逐 token 生成并投递；stop 置位后当前 token 完成即退出。"""
        error: Exception | None = None
        completion_tokens = 0
        try:
            with observe_inference("chat"):
                for chunk in llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=list(_STOP_TOKENS),
                    stream=True,
                ):
                    if stop.is_set():
                        return
                    completion_tokens += 1
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
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            _offer(loop, queue, error if error is not None else StreamEnd(usage), stop)

    worker = asyncio.create_task(asyncio.to_thread(generate))
    _active_workers.add(worker)
    worker.add_done_callback(_active_workers.discard)
    try:
        while True:
            item = await queue.get()
            if isinstance(item, StreamEnd):
                yield item
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        # 客户端断连（任务取消）或正常结束时通知生成线程停止
        stop.set()


# ===== 向量 =====


async def embed(llm: Any, texts: list[str]) -> tuple[list[list[float]], int]:
    """批量向量化：返回 (与输入同序的向量, 累计 prompt_tokens)"""
    vectors = []
    prompt_tokens = 0
    with observe_inference("embedding"):
        for text in texts:
            # 逐条持锁：长文档批量嵌入不长时间独占推理锁，对话请求可公平穿插
            async with _infer_lock:
                result = await asyncio.to_thread(llm.create_embedding, text)
            vectors.append(result["data"][0]["embedding"])
            prompt_tokens += result.get("usage", {}).get("prompt_tokens", 0)
    return vectors, prompt_tokens


# ===== 重排（Qwen3-Reranker 官方用法：首 token yes/no logprob 打分）=====

_RERANK_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
_RERANK_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


def _build_rerank_prompt(query: str, doc: str) -> str:
    """Qwen3-Reranker 官方 prompt 格式（ChatML，对齐官方 format_instruction）。

    assistant 后缀的空思考块 <think>\\n\\n</think> 为官方要求：占住思考生成位，
    使末位 logits 直接对应 yes/no 答案（缺它时模型实际想输出 <think>，分布失真）。
    不复用对话模板：那是对话模型模板（首条 system 注入 /no_think 软指令），
    reranker 需官方原始格式打分。
    """
    user = f"<Instruct>: {_RERANK_INSTRUCT}\n\n<Query>: {query}\n\n<Document>: {doc}"
    return (
        f"<|im_start|>system\n{_RERANK_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _yes_no_score(yes_logprob: float, no_logprob: float) -> float:
    """yes/no 二选一 softmax 后的 yes 概率。

    两 token 共享全词表 softmax 分母，比值只由二者 logit 差决定；
    减最大值再 exp 防 logit 较大时上溢。
    """
    if yes_logprob == -math.inf and no_logprob == -math.inf:
        return 0.0
    m = max(yes_logprob, no_logprob)
    p_yes = math.exp(yes_logprob - m)
    return p_yes / (p_yes + math.exp(no_logprob - m))


def _last_token_logits(llm: Any) -> Any:
    """eval 后末位 token 的全词表 logits（llama.cpp 总是输出末位 logits）。"""
    import llama_cpp
    import numpy as np

    ptr = llama_cpp.llama_get_logits_ith(llm.ctx, -1)
    return np.ctypeslib.as_array(ptr, shape=(llm.n_vocab(),))


def _token_id(llm: Any, token: str) -> int:
    ids = llm.tokenize(token.encode("utf-8"), add_bos=False, special=False)
    if len(ids) != 1:
        raise RuntimeError(f"rerank 打分依赖 {token!r} 为单 token，实际 ids={ids}")
    return ids[0]


def _score_pair(llm: Any, query: str, doc: str) -> float:
    """单对 (query, doc) 相关性打分 ∈ [0, 1]（Qwen3-Reranker 官方 Transformers 用法）。

    空思考块后取末位 logits 的 yes/no 二选一 softmax。不用
    create_completion(logprobs=...)：该参数要求 logits_all=True（16k ctx × 15 万
    词表的 logits 矩阵约 10GB 内存），低层 eval 仅输出末位 logits，零额外内存。
    """
    prompt = _build_rerank_prompt(query, doc)
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    # 预留分词误差余量，避免越 n_ctx 报错
    max_prompt_tokens = settings.CTX_SIZE - 4
    if len(tokens) > max_prompt_tokens:
        # 超长文档按 token 比例截断（0.9 吸收重新分词误差），保留头部（语义靠前）
        doc = doc[: int(len(doc) * max_prompt_tokens / len(tokens) * 0.9)]
        prompt = _build_rerank_prompt(query, doc)
        tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        logger.warning("rerank 文档超长已截断至 %d 字符（片段: %.60s…）", len(doc), doc)
    llm.reset()
    # reset 后再 eval：eval 只裁剪超界 KV，不清理跨 prompt 的残留上下文
    llm.eval(tokens)
    logits = _last_token_logits(llm)
    return _yes_no_score(logits[_token_id(llm, "yes")], logits[_token_id(llm, "no")])


async def rerank_scores(llm: Any, query: str, documents: list[str]) -> list[float]:
    """逐对打分：返回与 documents 同序的 relevance_score"""
    scores = []
    with observe_inference("rerank"):
        for doc in documents:
            # 逐对持锁：与对话/嵌入共用推理锁（llama.cpp 非线程安全），长文档不长时间独占
            async with _infer_lock:
                scores.append(await asyncio.to_thread(_score_pair, llm, query, doc))
    return scores


# 后台下载去重：首次 /v1/rerank 触发一次，失败后下次请求允许重试
_rerank_download_thread: threading.Thread | None = None
_rerank_download_lock = threading.Lock()


def trigger_rerank_download() -> None:
    """后台触发 rerank 模型下载（幂等；进行中不重复拉线程）。

    实际下载由 model_store 的 fcntl 文件锁跨进程互斥；本函数只保证单进程内
    不并发拉起多个下载线程。
    """
    global _rerank_download_thread
    with _rerank_download_lock:
        if _rerank_download_thread is not None and _rerank_download_thread.is_alive():
            return

        def _download():
            try:
                ensure_rerank_model()
            except Exception as exc:
                logger.warning("后台下载 rerank 模型失败（下次请求将重试）: %s", exc)

        _rerank_download_thread = threading.Thread(
            target=_download, name="rerank-model-download", daemon=True
        )
        _rerank_download_thread.start()
