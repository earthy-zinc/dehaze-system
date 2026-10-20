"""图像质量指标端点：逐指标隔离失败。"""

import asyncio
import logging

from fastapi import APIRouter, File, Form, UploadFile

from algorithm.metrics import METRICS_CONFIG, calculate_metrics
from app.errors import ModalError
from app.payload import open_image, parse_json, pixel_count

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["metrics"])


@router.post("/metrics", summary="计算图像质量指标")
async def compute_metrics(
    image: UploadFile = File(..., description="处理结果图字节"),
    model: str = Form(..., description="指标集标识（用于日志与幂等）"),
    metrics: str = Form(..., description="JSON 数组：指标名清单"),
    reference: UploadFile | None = File(None, description="可选参考图（GT）"),
) -> dict:
    if not model.strip():
        raise ModalError("PARAM_ERROR", "model 不能为空")

    names = parse_json(metrics, "metrics")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ModalError("PARAM_ERROR", "metrics 必须是非空的字符串数组")
    unsupported = [name for name in names if name not in METRICS_CONFIG]
    if unsupported:
        # 指标清单由编排侧按 /v1/models 下发，出现未知指标属契约错误，不静默降级
        raise ModalError("PARAM_ERROR", f"不支持的指标: {', '.join(unsupported)}")

    pil_image = open_image(image)
    reference_image = open_image(reference) if reference is not None else None
    if reference_image is not None and pil_image.size != reference_image.size:
        raise ModalError(
            "PARAM_ERROR",
            f"预测图与参考图尺寸不一致: 预测图{pil_image.size} vs 参考图{reference_image.size}",
        )

    logger.info("指标计算请求: model=%s, metrics=%s, image_size=%s", model, names, pil_image.size)
    # 指标计算无服务侧超时，上限由编排侧设定；逐指标隔离失败
    values, failed = await asyncio.to_thread(calculate_metrics, pil_image, reference_image, names)
    return {
        "metrics": values,
        "failed": failed,
        "meta": {
            "image_size": pixel_count(pil_image),
            "width": pil_image.size[0],
            "height": pil_image.size[1],
        },
    }
