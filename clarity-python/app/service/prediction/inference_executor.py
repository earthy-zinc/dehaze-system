"""图像推理执行：经统一适配器调用图像模型（本地 clarity-modal / 云厂商）。

编排侧保留 30s 超时与「超时即失败」语义（终态由 sys_pred_log 状态机裁决），推理本身
在引擎服务/云厂商执行——import_path/weights 等运行工件由引擎目录按 model_id 解析，
编排侧不再下发（设计 §2.4）。
"""

import logging

from app.infrastructure.llm.client import openai_image_client
from app.service.prediction.image_route import ImageRoute

logger = logging.getLogger(__name__)

# 单图推理超时阈值（秒）：超时任务标记失败（需求规格 §2.6.1/§9）。
# 引擎侧同阈值自限并返回 504，两条路径都按「超时即失败」处理。
INFERENCE_TIMEOUT_SECONDS = 30


async def run_edit(route: ImageRoute, image_bytes: bytes, params: dict | None = None) -> bytes:
    """调用图像模型执行编辑/去雾推理，返回结果图 PNG 字节。"""
    return await openai_image_client.edit(
        route.provider,
        route.model_id,
        image_bytes,
        params,
        timeout=INFERENCE_TIMEOUT_SECONDS,
    )
