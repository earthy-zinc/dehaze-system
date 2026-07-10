import logging
import traceback

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from jose import JWTError
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.core.code import ResultCode

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
        super().__init__(ResultCode.TASK_CANCELLED, "任务已被取消")


def register_exception_handlers(app: FastAPI):
    @app.exception_handler(BusinessException)
    async def business_exception_handler(request: Request, exc: BusinessException):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": exc.code.code,
                "msg": exc.message,
                "data": None,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        error_details = [
            {"field": ".".join(str(x) for x in e.get("loc", [])),
             "message": e.get("msg", "格式错误"),
             "code": e.get("type", "value_error")}
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

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": ResultCode.PARAM_ERROR.code,
                "msg": msg,
                "data": None,
                "errors": error_details,
            },
        )

    @app.exception_handler(JWTError)
    async def jwt_exception_handler(request: Request, exc: JWTError):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "code": ResultCode.TOKEN_INVALID.code,
                "msg": ResultCode.TOKEN_INVALID.msg,
                "data": None,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
        _logger.error(f"数据库异常: {exc}", exc_info=True)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code": ResultCode.DATABASE_ERROR.code,
                "msg": "数据库操作失败，请稍后重试",
                "data": None,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """通用异常（兜底）"""
        _logger.error(f"未处理异常: {exc}\n{traceback.format_exc()}")

        if settings.DEBUG:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "code": ResultCode.SYSTEM_EXECUTION_ERROR.code,
                    "msg": f"系统内部错误: {type(exc).__name__}",
                    "data": None,
                },
            )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code": ResultCode.SYSTEM_EXECUTION_ERROR.code,
                "msg": ResultCode.SYSTEM_EXECUTION_ERROR.msg,
                "data": None,
            },
        )
