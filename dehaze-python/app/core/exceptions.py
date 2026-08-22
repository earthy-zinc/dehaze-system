import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.core.code import ResultCode
from app.core.result import _get_trace_id
from app.middleware.non_null_response import NonNullJSONResponse as JSONResponse

_logger = logging.getLogger(__name__)


class BusinessException(Exception):
    """
    业务异常

    支持两种初始化方式：
    1. 使用 ResultCode 枚举：BusinessException(ResultCode.PARAM_ERROR)
    2. 使用字符串消息：BusinessException("错误信息")，将使用 SYSTEM_EXECUTION_ERROR 作为错误码
    """

    def __init__(self, code_or_message: ResultCode | str, message: str | None = None):
        if isinstance(code_or_message, str):
            # 如果第一个参数是字符串，使用默认错误码
            self.code = ResultCode.SYSTEM_EXECUTION_ERROR
            self.message = code_or_message
        else:
            # 如果第一个参数是 ResultCode
            self.code = code_or_message
            self.message = message or code_or_message.msg


class TaskCancelledException(BusinessException):
    """任务取消异常（专用于任务执行中检测到取消标志时抛出）"""

    def __init__(self):
        super().__init__(ResultCode.TASK_CANCELLED)


def register_exception_handlers(app: FastAPI):
    @app.exception_handler(BusinessException)
    async def business_exception_handler(request: Request, exc: BusinessException):
        request.state.db_should_rollback = True
        _logger.error("业务异常: %s", exc.message, extra={"code": exc.code.code, "status": 400})
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": exc.code.code,
                "msg": exc.message,
                "data": None,
                "traceId": _get_trace_id(),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        error_details = [
            {
                "field": ".".join(str(x) for x in e.get("loc", [])),
                "message": e.get("msg", "格式错误"),
                "code": e.get("type", "value_error"),
            }
            for e in errors
        ]
        if errors:
            first_error = errors[0]
            msg = f"参数校验失败: {first_error.get('msg', '未知错误')}"
            if "loc" in first_error:
                loc = ".".join(str(x) for x in first_error["loc"])
                msg = f"参数 {loc} {first_error.get('msg', '格式错误')}"
        else:
            msg = "请求参数格式错误"

        _logger.warning(
            "参数校验失败: %s", msg, extra={"code": ResultCode.PARAM_ERROR.code, "status": 400}
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": ResultCode.PARAM_ERROR.code,
                "msg": msg,
                "data": None,
                "errors": error_details,
                "traceId": _get_trace_id(),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request.state.db_should_rollback = True
        # 根据 HTTP 状态码映射到对应的 ResultCode
        status_code_map = {
            status.HTTP_401_UNAUTHORIZED: ResultCode.TOKEN_INVALID,
            status.HTTP_403_FORBIDDEN: ResultCode.ACCESS_UNAUTHORIZED,
            status.HTTP_400_BAD_REQUEST: ResultCode.PARAM_ERROR,
            status.HTTP_404_NOT_FOUND: ResultCode.RESOURCE_NOT_FOUND,
        }
        result_code = status_code_map.get(exc.status_code, ResultCode.SYSTEM_EXECUTION_ERROR)

        _logger.error(
            "HTTP 异常: %s",
            result_code.msg,
            extra={"code": result_code.code, "status": exc.status_code},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": result_code.code,
                "msg": result_code.msg,
                "data": None,
                "traceId": _get_trace_id(),
            },
            headers=exc.headers,
        )

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
        request.state.db_should_rollback = True
        _logger.error(
            "数据库异常: %s",
            exc,
            extra={"code": ResultCode.DATABASE_ERROR.code, "status": 500},
            exc_info=True,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code": ResultCode.DATABASE_ERROR.code,
                "msg": "数据库操作失败，请稍后重试",
                "data": None,
                "traceId": _get_trace_id(),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """通用异常（兜底）"""
        request.state.db_should_rollback = True
        _logger.error(
            "未处理异常: %s",
            exc,
            extra={"code": ResultCode.SYSTEM_EXECUTION_ERROR.code, "status": 500},
            exc_info=True,
        )

        if settings.DEBUG:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "code": ResultCode.SYSTEM_EXECUTION_ERROR.code,
                    "msg": f"系统内部错误: {type(exc).__name__}",
                    "data": None,
                    "traceId": _get_trace_id(),
                },
            )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code": ResultCode.SYSTEM_EXECUTION_ERROR.code,
                "msg": ResultCode.SYSTEM_EXECUTION_ERROR.msg,
                "data": None,
                "traceId": _get_trace_id(),
            },
        )
