"""错误模型：错误响应统一 {"code","message"}，HTTP 状态承载处置语义。

引擎侧错误在产生点定型（400 参数非法、422 音频非法、503 模型不可用、500 推理失败），
`code` 与编排侧 ResultCode 的映射由编排侧客户端集中定义，服务侧不做透传。
5xx 响应一律不携带异常原文（原文只进日志），避免把调用栈/文件路径暴露给调用方。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class VoiceError(Exception):
    """引擎侧业务错误：code 与 HTTP 状态由子类定型，message 由产生点给出"""

    code = "INTERNAL_ERROR"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ParamError(VoiceError):
    """请求参数缺失/格式错误（编排侧不重试，转 PARAM_ERROR）"""

    code = "PARAM_ERROR"
    status_code = 400


class AudioInvalidError(VoiceError):
    """音频内容非法（编码/声道/位深/采样率不符规格）"""

    code = "AUDIO_INVALID"
    status_code = 422


class ModelUnavailableError(VoiceError):
    """依赖未安装或模型/权重不可用（编排侧有限重试）"""

    code = "MODEL_UNAVAILABLE"
    status_code = 503


class InferenceError(VoiceError):
    """推理执行失败（编排侧按业务失败路径处理，不重试）"""

    code = "INFERENCE_FAILED"
    status_code = 500


# 非 /v1 语义的 HTTP 状态：保留状态码语义，只把响应体归一为错误模型
_HTTP_CODES = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(VoiceError)
    async def _voice_error(request: Request, exc: VoiceError) -> JSONResponse:
        logger.warning(
            "引擎请求失败 code=%s status=%s message=%s",
            exc.code,
            exc.status_code,
            exc.message,
            extra={"code": exc.code, "status": exc.status_code},
        )
        return JSONResponse({"code": exc.code, "message": exc.message}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI 校验失败属参数非法：message 保留字段级信息供编排侧定位
        message = "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        logger.warning(
            "请求参数校验失败 path=%s message=%s",
            request.url.path,
            message,
            extra={"code": "PARAM_ERROR", "status": 400},
        )
        return JSONResponse({"code": "PARAM_ERROR", "message": message}, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error(
                "HTTP 异常 path=%s status=%s detail=%s",
                request.url.path,
                exc.status_code,
                exc.detail,
                exc_info=exc,
                extra={"code": "INTERNAL_ERROR", "status": exc.status_code},
            )
            return JSONResponse(
                {"code": "INTERNAL_ERROR", "message": "内部错误"}, status_code=exc.status_code
            )
        code = _HTTP_CODES.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse({"code": code, "message": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "未处理异常 path=%s error=%s",
            request.url.path,
            exc,
            exc_info=exc,
            extra={"code": "INTERNAL_ERROR", "status": 500},
        )
        return JSONResponse({"code": "INTERNAL_ERROR", "message": "内部错误"}, status_code=500)
