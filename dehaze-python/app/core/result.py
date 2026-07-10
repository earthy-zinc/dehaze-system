from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from app.core.code import ResultCode

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """参数校验错误项"""
    field: str = Field(description="字段名")
    message: str = Field(description="错误消息")
    code: str = Field(description="校验规则码")


class Result(BaseModel, Generic[T]):
    """统一响应结构（与 Java/Go 保持一致）"""
    code: str = ResultCode.SUCCESS.code
    msg: str = ResultCode.SUCCESS.msg
    data: Optional[T] = None
    traceId: Optional[str] = Field(default=None, serialization_alias="traceId")
    timestamp: Optional[int] = Field(default=None)
    errors: Optional[list] = Field(default=None)


def success(data: Any = None, msg: str = ResultCode.SUCCESS.msg) -> Result:
    return Result(code=ResultCode.SUCCESS.code, msg=msg, data=data)


def error(msg: str, code: str = ResultCode.SYSTEM_EXECUTION_ERROR.code) -> Result:
    return Result(code=code, msg=msg, data=None)


def warning(code: ResultCode) -> Result:
    return Result(code=code.code, msg=code.msg, data=None)


def success_response(data: Any = None, msg: str = ResultCode.SUCCESS.msg) -> dict:
    return {"code": ResultCode.SUCCESS.code, "msg": msg, "data": data}


def error_response(msg: str, code: str = ResultCode.SYSTEM_EXECUTION_ERROR.code) -> dict:
    return {"code": code, "msg": msg, "data": None}
