"""算法推理执行：专用线程池（并发数可配）+ dehaze 算法调用。"""

import asyncio
import importlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor

import PIL.Image

from algorithm.model_loader import resolve_model_path
from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException

logger = logging.getLogger(__name__)

# 算法推理专用线程池：PyTorch 推理为 CPU 密集型同步操作，
# 必须在线程池中执行以避免阻塞 asyncio 事件循环。
# 并发数通过 INFERENCE_THREAD_POOL_SIZE 配置，按 GPU 显存/卡数调整。
_inference_executor = ThreadPoolExecutor(
    max_workers=settings.INFERENCE_THREAD_POOL_SIZE, thread_name_prefix="algo-inference"
)

# 单图推理超时阈值（秒）：超时任务标记失败并回滚配额（需求规格 §2.6.1/§9）。
INFERENCE_TIMEOUT_SECONDS = 30


async def run_dehaze(
    import_path: str, model_relative_path: str, image_bytes: io.BytesIO
) -> io.BytesIO:
    """在线程池中执行去雾推理（避免阻塞事件循环），超时标记失败。

    线程池中的同步推理无法被强制中断，超时后放弃等待结果，任务按失败处理。
    """
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                _inference_executor, _run_dehaze_sync, import_path, model_relative_path, image_bytes
            ),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning(
            "推理超时(%ss)，任务标记失败: module=%s", INFERENCE_TIMEOUT_SECONDS, import_path
        )
        raise BusinessException(
            ResultCode.SYSTEM_EXECUTION_ERROR,
            f"任务执行超时（{INFERENCE_TIMEOUT_SECONDS}s），建议更换算法或降低图片分辨率",
        ) from None


def _run_dehaze_sync(
    import_path: str, model_relative_path: str, image_bytes: io.BytesIO
) -> io.BytesIO:
    """
    调用算法去雾

    Args:
        import_path: 算法模块导入路径，如 'algorithm.AECRNet.run'，仅用于 importlib
        model_relative_path: 模型权重文件相对路径（sys_algorithm.path），
                             如 'AECR-Net/NH_train.pk'，用于通过 model_loader 解析本地路径
    """
    # import_path 仅用于 importlib，不再反推文件目录
    module_name = import_path
    if module_name.startswith("algorithm."):
        module_name = module_name[len("algorithm.") :]
    if module_name.endswith(".run"):
        module_name = module_name[: -len(".run")]

    try:
        algo_module = importlib.import_module(f"algorithm.{module_name}.run")
    except ImportError as e:
        raise BusinessException(
            ResultCode.SYSTEM_EXECUTION_ERROR,
            f"算法模块加载失败: algorithm.{module_name}.run, "
            f"请确认 import_path '{import_path}' 是否正确. "
            f"原始错误: {e}",
        ) from None

    if not hasattr(algo_module, "dehaze"):
        raise BusinessException(
            ResultCode.SYSTEM_EXECUTION_ERROR,
            f"算法模块 {module_name} 未导出 dehaze() 函数",
        )

    dehaze_fn = algo_module.dehaze

    # 通过 model_loader 解析模型权重文件到本地路径
    # 算法 path 字段为空（如 DCP 无权重）时传空字符串，由算法自行处理
    model_path = ""
    if model_relative_path and model_relative_path.strip():
        try:
            model_path = resolve_model_path(model_relative_path)
        except FileNotFoundError as e:
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR,
                f"模型权重加载失败: {e}",
            ) from e

    logger.debug("执行去雾: module=%s, model=%s", module_name, model_path)

    # 调用 dehaze 函数（算法内部自行加载权重；异常统一包装为业务错误，
    # 截取摘要避免泄露绝对路径/完整堆栈）
    try:
        result = dehaze_fn(image_bytes, model_path)
    except BusinessException:
        raise
    except Exception as e:
        raise BusinessException(
            ResultCode.SYSTEM_EXECUTION_ERROR,
            f"算法执行失败: {module_name} - {str(e)[:200]}",
        ) from e

    if isinstance(result, io.BytesIO):
        return result
    if isinstance(result, PIL.Image.Image):
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        return buf
    raise BusinessException(
        ResultCode.SYSTEM_EXECUTION_ERROR,
        f"dehaze() 返回了不支持的类型: {type(result)}",
    )
