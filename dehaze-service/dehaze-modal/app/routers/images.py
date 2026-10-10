"""OpenAI Images 兼容端点（阶段 3）：`POST /v1/images/edits`。

字段对齐 OpenAI Images edit：multipart `image` + `model`；`model` 取统一模型表的
规范 slug（如 `aecrnet`），import_path/weights 由服务端目录（app.catalog）解析，
编排侧不再下发运行工件。响应按 OpenAI Images 风格返回 `{created, data:[{b64_json}]}`
（JSON b64 较裸字节 +33% 载荷，内网可接受）。
"""

import base64
import logging
import time

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app import catalog
from app.errors import ModalError
from app.executor import run_inference
from app.payload import open_image, parse_json, pixel_count

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/images", tags=["images"])


@router.post(
    "/edits",
    response_class=JSONResponse,
    summary="图像编辑（去雾等）：按模型 slug 推理，结果以 b64 返回",
)
async def images_edits(
    image: UploadFile = File(..., description="原图字节"),
    model: str = Form(..., description="模型标识（统一模型表规范 slug，如 aecrnet）"),
    params: str | None = Form(None, description="可选 JSON，推理参数（当前模型算法均无该入参）"),
) -> JSONResponse:
    if not model.strip():
        raise ModalError("PARAM_ERROR", "model 不能为空")
    item = catalog.resolve_edit(model.strip())

    if params is not None:
        # 契约字段：现状模型算法签名均为 dehaze(image, model_path)，不接受额外参数，
        # 故仅校验格式并留痕，不向算法透传（透传会破坏算法调用签名）
        parsed = parse_json(params, "params")
        if not isinstance(parsed, dict):
            raise ModalError("PARAM_ERROR", "params 必须是 JSON 对象")

    pil_image = open_image(image)
    image_size = pixel_count(pil_image)
    image.file.seek(0)

    logger.info(
        "图像编辑请求: model=%s, import_path=%s, weights=%s, image_size=%s",
        model,
        item["import_path"],
        item["weights_path"] or "-",
        image_size,
    )
    result = await run_inference(
        model, item["import_path"], item["weights_path"], image.file, image_size
    )
    return JSONResponse(
        {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(result).decode("ascii")}],
        }
    )
