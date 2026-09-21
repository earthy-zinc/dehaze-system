"""本地轻量 LLM 模型文件自动下载（chat 模型 + embedding 模型 + rerank 模型）

部署零手工：模型文件不存在时自动下载（懒下载 + 启动后台预下载双触发）。

- 镜像回退：hf-mirror.com → huggingface.co
- 断点续传：.part 临时文件 + HTTP Range 请求接续已下载字节
- 完整性校验：最终字节数与远端 x-linked-size 比对后原子 rename 生效
- 跨进程互斥：fcntl 文件锁（多 worker 并发触发时仅一个进程实际下载，其余等待；
  Windows 无 fcntl，开发环境单进程退化为无锁直下）
"""

try:
    import fcntl
except ImportError:  # Windows 无 fcntl
    fcntl = None
import logging
import time
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# 对话模型：unsloth/Qwen3-0.6B-GGUF（内容与 Qwen 官方权重同源，Apache-2.0）
MODEL_FILE = "Qwen3-0.6B-Q4_K_M.gguf"
EXPECTED_SIZE = 396705472  # 字节（远端 x-linked-size，Q4_K_M 量化）
_DOWNLOAD_URLS = [
    f"https://hf-mirror.com/unsloth/Qwen3-0.6B-GGUF/resolve/main/{MODEL_FILE}",
    f"https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/{MODEL_FILE}",
]

# 向量模型：Qwen 官方 Qwen3-Embedding-0.6B（1024 维，与 bge-m3 同维度）
EMBEDDING_MODEL_FILE = "Qwen3-Embedding-0.6B-Q8_0.gguf"
EMBEDDING_EXPECTED_SIZE = 639150592
_EMBEDDING_URLS = [
    f"https://hf-mirror.com/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/{EMBEDDING_MODEL_FILE}",
    f"https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/{EMBEDDING_MODEL_FILE}",
]

# 重排模型：Qwen3-Reranker-0.6B（Q8_0，约 610MB；rerank 打分对精度敏感故不用 Q4）。
# Qwen 官方未发布该模型 GGUF，Mungert 仓库为 llama.cpp 官方转换脚本产物（社区通用）；
# 文件大小为 hf-mirror / huggingface 双源 HEAD 实测 Content-Length（639150432 字节）
RERANK_MODEL_FILE = "Qwen3-Reranker-0.6B-q8_0.gguf"
RERANK_EXPECTED_SIZE = 639150432
_RERANK_URLS = [
    f"https://hf-mirror.com/Mungert/Qwen3-Reranker-0.6B-GGUF/resolve/main/{RERANK_MODEL_FILE}",
    f"https://huggingface.co/Mungert/Qwen3-Reranker-0.6B-GGUF/resolve/main/{RERANK_MODEL_FILE}",
]

_CHUNK = 1024 * 1024  # 1MB
_PROGRESS_STEP = 0.1


def model_path() -> str:
    """对话模型路径（仓库根 models/，不入 git；可经 LOCAL_LLM_MODEL_PATH 覆盖）"""
    configured = settings.LOCAL_LLM_MODEL_PATH.strip()
    return configured if configured else str(Path(settings.MODEL_CACHE_DIR) / MODEL_FILE)


def embedding_model_path() -> str:
    """向量模型路径（可经 LOCAL_LLM_EMBEDDING_MODEL_PATH 覆盖）"""
    configured = settings.LOCAL_LLM_EMBEDDING_MODEL_PATH.strip()
    return configured if configured else str(Path(settings.MODEL_CACHE_DIR) / EMBEDDING_MODEL_FILE)


def rerank_model_path() -> str:
    """重排模型路径（可经 LOCAL_LLM_RERANK_MODEL_PATH 覆盖）"""
    configured = settings.LOCAL_LLM_RERANK_MODEL_PATH.strip()
    return configured if configured else str(Path(settings.MODEL_CACHE_DIR) / RERANK_MODEL_FILE)


def is_downloaded() -> bool:
    path = Path(model_path())
    return path.exists() and path.stat().st_size == EXPECTED_SIZE


def is_embedding_downloaded() -> bool:
    path = Path(embedding_model_path())
    return path.exists() and path.stat().st_size == EMBEDDING_EXPECTED_SIZE


def is_rerank_downloaded() -> bool:
    path = Path(rerank_model_path())
    return path.exists() and path.stat().st_size == RERANK_EXPECTED_SIZE


def ensure_model() -> str:
    """确保对话模型就绪，返回路径；已就绪直接返回，否则下载（跨进程互斥）。"""
    return _ensure(model_path(), EXPECTED_SIZE, _DOWNLOAD_URLS)


def ensure_embedding_model() -> str:
    """确保向量模型就绪，返回路径。"""
    return _ensure(embedding_model_path(), EMBEDDING_EXPECTED_SIZE, _EMBEDDING_URLS)


def ensure_rerank_model() -> str:
    """确保重排模型就绪，返回路径。

    rerank 为可选功能：manager 启动/lifecycle 预下载刻意不触发（避免不用 rerank
    的部署白下 610MB），由 local_llm_server 的 /v1/rerank 首次请求后台调用，
    或部署方手动预下载：python -c "from app.infrastructure.llm.local.local_llm_model
    import ensure_rerank_model; ensure_rerank_model()"
    """
    return _ensure(rerank_model_path(), RERANK_EXPECTED_SIZE, _RERANK_URLS)


def _ensure(path: str, expected_size: int, urls: list[str]) -> str:
    if _file_ready(path, expected_size):
        return path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not fcntl:  # Windows：无锁直下
        _download(path, expected_size, urls)
        return path
    with Path(path + ".lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if _file_ready(path, expected_size):  # 等锁期间其他进程已完成
                return path
            _download(path, expected_size, urls)
            return path
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _file_ready(path: str, expected_size: int) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size == expected_size


def _download(path: str, expected_size: int, urls: list[str]) -> None:
    part = path + ".part"
    last_error: Exception | None = None
    for url in urls:
        try:
            _download_from(url, part, expected_size)
            _finalize(part, path, expected_size)
            return
        except Exception as exc:
            last_error = exc
            logger.warning("模型下载失败（%s）：%s，尝试下一个镜像", url, exc)
    raise RuntimeError(
        f"本地模型自动下载失败（{Path(path).name}，{expected_size // 1048576}MB）：{last_error}。"
        "请检查网络可达 hf-mirror.com / huggingface.co，或手动下载放置于 models/ 目录"
    )


def _download_from(url: str, part: str, expected_size: int) -> None:
    part_path = Path(part)
    offset = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with (
        httpx.Client(
            timeout=httpx.Timeout(connect=15, read=120, write=30, pool=15), follow_redirects=True
        ) as client,
        client.stream("GET", url, headers=headers) as resp,
    ):
        if offset and resp.status_code != 206:
            offset = 0  # 服务端不支持续传（返回 200 全量）→ 重头下载
        resp.raise_for_status()
        total = offset + int(resp.headers.get("content-length", 0))
        if total and total != expected_size:
            raise RuntimeError(f"远端文件大小 {total} 与预期 {expected_size} 不一致")

        mode = "ab" if offset else "wb"
        downloaded = offset
        next_log = (downloaded / expected_size) + _PROGRESS_STEP if expected_size else 1.0
        started = time.perf_counter()
        with part_path.open(mode) as f:
            for chunk in resp.iter_bytes(chunk_size=_CHUNK):
                f.write(chunk)
                downloaded += len(chunk)
                progress = downloaded / expected_size if expected_size else 0
                if progress >= next_log:
                    next_log += _PROGRESS_STEP
                    speed = (
                        (downloaded - offset) / max(time.perf_counter() - started, 0.1) / 1048576
                    )
                    logger.info(
                        "模型下载中 %.0f%%（%d/%dMB，%.1fMB/s）",
                        progress * 100,
                        downloaded // 1048576,
                        expected_size // 1048576,
                        speed,
                    )
        if downloaded != expected_size:
            raise RuntimeError(f"下载数不完整：{downloaded} != {expected_size}")


def _finalize(part: str, path: str, expected_size: int) -> None:
    size = Path(part).stat().st_size
    if size != expected_size:
        raise RuntimeError(f"校验失败：下载数 {size} != 预期 {expected_size}")
    Path(part).replace(path)  # 原子生效
    logger.info("本地模型就绪：%s（%dMB）", path, size // 1048576)
