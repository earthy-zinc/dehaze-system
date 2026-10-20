"""错误模型。

错误响应统一为 {"code": "<大写下划线常量>", "message": "<可读信息>"}；
状态码与错误码的对应关系集中在此，避免调用点各自硬编码。
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

_CODE_STATUS = {
    "PARAM_ERROR": 400,
    "UNAUTHORIZED": 401,
    "NOT_FOUND": 404,
    # 统一模型管理阶段 3：模型目录（/v1/images/edits）的模型级错误
    "MODEL_NOT_FOUND": 404,
    "MODEL_DISABLED": 404,
    "INVALID_IMAGE": 422,
    "INFERENCE_FAILED": 500,
    "INTERNAL_ERROR": 500,
    "MODEL_UNAVAILABLE": 503,
    "SERVICE_UNAVAILABLE": 503,
    "INFERENCE_TIMEOUT": 504,
}


class ModalError(Exception):
    """带错误码的服务异常，由全局异常处理器转为统一错误响应。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.status_code = _CODE_STATUS[code]
        self.message = message


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ModalError)
    async def _modal_error_handler(_: Request, exc: ModalError) -> JSONResponse:
        log = logger.error if exc.status_code >= 500 else logger.warning
        log("服务异常: %s", exc.message, extra={"code": exc.code, "status": exc.status_code})
        return JSONResponse(
            status_code=exc.status_code, content={"code": exc.code, "message": exc.message}
        )

    # 请求体/表单字段缺失或格式非法属参数错误（400），422 保留给图像内容非法
    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(part) for part in detail.get("loc", ()) if part != "body")
        message = f"请求参数非法: {field or 'body'} - {detail.get('msg', '')}"
        logger.warning("服务异常: %s", message, extra={"code": "PARAM_ERROR", "status": 400})
        return JSONResponse(status_code=400, content={"code": "PARAM_ERROR", "message": message})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            code = "NOT_FOUND"
        elif 400 <= exc.status_code < 500:
            code = "PARAM_ERROR"
        else:
            code = "INTERNAL_ERROR"
        log = logger.error if exc.status_code >= 500 else logger.warning
        log("服务异常: %s", exc.detail, extra={"code": code, "status": exc.status_code})
        return JSONResponse(
            status_code=exc.status_code, content={"code": code, "message": str(exc.detail)}
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        # 异常原文只进日志（含堆栈），响应固定文案：避免把内部实现细节/路径外泄给调用方
        logger.exception("未处理异常: %s", exc, extra={"code": "INTERNAL_ERROR", "status": 500})
        return JSONResponse(
            status_code=500, content={"code": "INTERNAL_ERROR", "message": "内部错误"}
        )
