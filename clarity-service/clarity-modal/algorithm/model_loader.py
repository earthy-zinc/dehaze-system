"""模型权重文件加载器。

权重统一托管在 nginx 静态服务（`MODEL_BASE_URL`，对应 /models 路径），服务侧按
`MODEL_CACHE_DIR/{relative_path}` 缓存后交给 `torch.load` 读取。

加载流程：
1. `MODEL_CACHE_DIR/{relative_path}` 命中 → 直接返回本地路径（不进网络，即为降级口径）
2. 未命中 → 从 `MODEL_BASE_URL/{relative_path}` 下载到缓存路径
3. 下载失败 → 报 `FileNotFoundError`，由调用方转 503（模型权重不可用）

并发去重：按 relative_path 维度加锁，同一权重同时仅允许一个下载请求，其他等待者读缓存。

调用约定：服务侧在推理线程池内同步调用，故使用 threading.Lock 而非 asyncio.Lock。
"""

import logging
import threading
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# 按 relative_path 维度的下载锁，避免并发重复下载同一文件
_path_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _get_path_lock(relative_path: str) -> threading.Lock:
    with _locks_guard:
        lock = _path_locks.get(relative_path)
        if lock is None:
            lock = threading.Lock()
            _path_locks[relative_path] = lock
        return lock


def _build_url(relative_path: str) -> str:
    return f"{settings.MODEL_BASE_URL.rstrip('/')}/{relative_path.lstrip('/')}"


def _download_to_cache(url: str, cache_path: Path) -> None:
    """同步下载到缓存路径。

    先写临时文件再原子替换：写盘中断（磁盘满/进程被杀）不会在缓存目录留下
    半截文件——半截文件会被后续请求当作缓存命中，造成持续性的权重损坏。
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f"{cache_path.name}.part")
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        temp_path.write_bytes(response.content)
    temp_path.replace(cache_path)


def resolve_model_path(relative_path: str) -> str:
    """解析权重到本地可读路径（相对路径由图像模型目录内嵌，如 AECR-Net/NH_train.pk）。"""
    if not relative_path or not relative_path.strip():
        raise FileNotFoundError("模型路径为空，无法加载权重")

    cache_path = Path(settings.MODEL_CACHE_DIR) / relative_path
    if cache_path.exists():
        return str(cache_path)

    with _get_path_lock(relative_path):
        if cache_path.exists():
            return str(cache_path)

        url = _build_url(relative_path)
        try:
            _download_to_cache(url, cache_path)
        except Exception as e:
            raise FileNotFoundError(
                f"模型权重文件不可用: path={relative_path}, url={url}, error={e}"
            ) from e
        logger.info("模型权重下载成功: %s -> %s", url, cache_path)
        return str(cache_path)


def check_model_exists(relative_path: str) -> int | None:
    """校验权重可访问并返回字节数；不可访问返回 None（供权重可用性探测与体积展示）。"""
    if not relative_path or not relative_path.strip():
        return None

    cache_path = Path(settings.MODEL_CACHE_DIR) / relative_path
    if cache_path.exists():
        return cache_path.stat().st_size

    url = _build_url(relative_path)
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.head(url)
            if response.status_code != 200:
                return None
            content_length = response.headers.get("Content-Length")
            return int(content_length) if content_length is not None else None
    except Exception as e:
        logger.warning("模型权重校验失败: %s - %s", url, e)
        return None
