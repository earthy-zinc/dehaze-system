"""算法推理执行：线程池 + importlib 装载 + 30s 超时。

每次推理重新加载权重，不做模型实例缓存。
"""

import asyncio
import importlib
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import BinaryIO

import PIL.Image

from algorithm.model_loader import resolve_model_path
from app.config import settings
from app.errors import ModalError
from app.metrics import record_inference_metrics

logger = logging.getLogger(__name__)

# PyTorch 推理是 CPU 密集型同步调用，必须在线程池中执行以免阻塞事件循环
_executor = ThreadPoolExecutor(
    max_workers=settings.INFERENCE_THREAD_POOL_SIZE, thread_name_prefix="modal-inference"
)


async def run_inference(
    model: str,
    import_path: str,
    weights_path: str,
    image: BinaryIO,
    image_size: int | None,
) -> bytes:
    """在线程池中执行推理并返回结果图字节。

    线程池中的同步推理无法被强制中断：超时后放弃等待结果（线程继续跑完自行退出），
    按 504 返回，终态由编排侧状态机裁决。
    """
    start = time.monotonic()
    status = "success"
    try:
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, _execute, import_path, weights_path, image),
            timeout=settings.INFERENCE_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        status = "timeout"
        logger.warning(
            "推理超时(%ss): model=%s, import_path=%s",
            settings.INFERENCE_TIMEOUT_SECONDS,
            model,
            import_path,
        )
        raise ModalError(
            "INFERENCE_TIMEOUT",
            f"推理超时（{settings.INFERENCE_TIMEOUT_SECONDS:.0f}s），建议更换模型或降低图片分辨率",
        ) from None
    except Exception:
        status = "error"
        raise
    finally:
        record_inference_metrics(model, time.monotonic() - start, status, image_size)


def _execute(import_path: str, weights_path: str, image: BinaryIO) -> bytes:
    module = _load_algorithm_module(import_path)
    dehaze_fn = getattr(module, "dehaze", None)
    if not callable(dehaze_fn):
        raise ModalError("NOT_FOUND", f"算法模块未导出 dehaze(): {import_path}")

    # 算法 path 为空（如 DCP 无权重）时传空字符串，由算法自行处理
    model_path = ""
    if weights_path.strip():
        try:
            model_path = resolve_model_path(weights_path)
        except FileNotFoundError as e:
            raise ModalError("MODEL_UNAVAILABLE", f"模型权重不可用: {e}") from None

    try:
        result = dehaze_fn(image, model_path)
    except Exception as e:
        # 只回摘要，避免把绝对路径与完整堆栈透出给编排侧
        raise ModalError("INFERENCE_FAILED", f"算法执行失败: {import_path} - {str(e)[:200]}") from e

    if isinstance(result, io.BytesIO):
        return result.getvalue()
    if isinstance(result, PIL.Image.Image):
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return buffer.getvalue()
    raise ModalError("INFERENCE_FAILED", f"dehaze() 返回了不支持的类型: {type(result)}")


def _load_algorithm_module(import_path: str):
    """按 import_path 装载算法模块（如 algorithm.AECRNet.run）。

    import_path 由服务端模型目录（app/catalog.py）按模型 slug 解析，编排侧只下发 slug。
    """
    module_name = import_path.removeprefix("algorithm.").removesuffix(".run")
    target = f"algorithm.{module_name}.run"
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as e:
        # 缺失的是算法模块自身 → 标识不存在；缺失的是第三方依赖 → 服务环境问题
        if e.name == target or target.startswith(f"{e.name}."):
            raise ModalError("NOT_FOUND", f"算法不存在: {import_path}") from None
        raise ModalError("INFERENCE_FAILED", f"算法依赖缺失: {e.name}") from None
    except ImportError as e:
        raise ModalError("INFERENCE_FAILED", f"算法模块加载失败: {target} - {e}") from None
