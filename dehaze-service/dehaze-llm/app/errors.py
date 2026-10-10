"""错误模型：错误响应统一 {"code": "<大写下划线常量>", "message": "<可读信息>"}。

状态码语义：400 请求非法 / 404 标识不存在 / 500 推理执行失败 / 503 未就绪或模型不可用。
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """服务内可预期的失败：携带 HTTP 状态码与错误常量，由异常处理器统一序列化。"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def unauthorized_response() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"code": "UNAUTHORIZED", "message": "未授权：内部服务密钥缺失或错误"},
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"code": exc.code, "message": exc.message}
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        message = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        return JSONResponse(
            status_code=400, content={"code": "INVALID_REQUEST", "message": message}
        )

    # FastAPI/Starlette 自身抛出的 HTTPException（未知路由 404、方法不允许 405 等）必须同样
    # 走 {code,message} 信封，否则 FastAPI 默认的 {"detail": ...} 会绕过统一错误模型
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            code = "NOT_FOUND"
        elif 400 <= exc.status_code < 500:
            code = "INVALID_REQUEST"
        else:
            code = "INFERENCE_FAILED"
        if exc.status_code >= 500:
            logger.error("HTTP 异常: %s", exc.detail, exc_info=True)
        return JSONResponse(
            status_code=exc.status_code, content={"code": code, "message": str(exc.detail)}
        )

    @app.exception_handler(Exception)
    async def _unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        # 契约要求任何失败都走统一错误模型；未预期异常必须留痕，否则只能看到一句通用提示
        logger.error("未预期异常: %r", exc, exc_info=True)
        return JSONResponse(
            status_code=500, content={"code": "INFERENCE_FAILED", "message": "推理执行失败"}
        )
