"""
模型权重文件加载器

将算法权重文件统一纳入 Nginx 静态文件服务（nginx-dataset 容器 /models 路径），
Python 服务通过 HTTP 下载到本地缓存后由 torch.load 加载，解除与 trained_model/ 目录的强耦合。

加载流程：
1. 优先检查 MODEL_CACHE_DIR/{relative_path} 是否存在 → 命中则直接返回本地路径
2. 未命中则从 MODEL_BASE_URL/{relative_path} 下载到 MODEL_CACHE_DIR/{relative_path}
3. 下载失败时若 MODEL_FALLBACK_TO_LOCAL=True 且本地存在缓存，降级使用缓存

并发去重：按 relative_path 维度加锁，同一模型同时仅允许一个下载请求，其他等待者直接读缓存。

调用约定：
- 预测服务在 thread executor 中调用 resolve_model_path（同步）
- 算法模块（如 RIDCP/ITBdehaze）在构造模型时调用 resolve_model_path（同步）
- 因此使用 threading.Lock 而非 asyncio.Lock
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
    """同步 HTTP 下载文件到缓存路径（父目录自动创建）"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        cache_path.write_bytes(response.content)


def resolve_model_path(relative_path: str) -> str:
    """
    解析模型权重文件到本地可读路径。

    Args:
        relative_path: 相对路径，对应 sys_algorithm.path 字段，如 "AECR-Net/NH_train.pk"

    Returns:
        本地文件绝对路径字符串

    Raises:
        FileNotFoundError: 路径为空，或下载失败且本地无缓存
    """
    if not relative_path or not relative_path.strip():
        raise FileNotFoundError("模型路径为空，无法加载权重")

    cache_path = Path(settings.MODEL_CACHE_DIR) / relative_path

    # 快速路径：缓存命中
    if cache_path.exists():
        return str(cache_path)

    # 慢速路径：加锁下载，避免并发重复下载
    with _get_path_lock(relative_path):
        if cache_path.exists():
            return str(cache_path)

        url = _build_url(relative_path)
        try:
            _download_to_cache(url, cache_path)
            logger.info("模型权重下载成功: %s -> %s", url, cache_path)
            return str(cache_path)
        except Exception as e:
            if settings.MODEL_FALLBACK_TO_LOCAL and cache_path.exists():
                logger.warning(
                    "模型权重下载失败，降级使用本地缓存: %s - %s", relative_path, e
                )
                return str(cache_path)
            raise FileNotFoundError(
                f"模型权重文件不可用: path={relative_path}, url={url}, error={e}"
            ) from e


def check_model_exists(relative_path: str) -> int | None:
    """
    校验模型权重文件是否可访问，并返回文件字节数。

    用于算法 CRUD 时校验 path 字段有效性并回填 size 字段。

    Returns:
        文件字节数（int）；不可访问时返回 None
    """
    if not relative_path or not relative_path.strip():
        return None

    # 本地缓存命中：直接读文件大小
    cache_path = Path(settings.MODEL_CACHE_DIR) / relative_path
    if cache_path.exists():
        return cache_path.stat().st_size

    # 否则通过 HTTP HEAD 校验，并从 Content-Length 取字节数
    url = _build_url(relative_path)
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.head(url)
            if response.status_code != 200:
                return None
            content_length = response.headers.get("Content-Length")
            if content_length is None:
                return None
            return int(content_length)
    except Exception as e:
        logger.warning("模型权重校验失败: %s - %s", url, e)
        return None
