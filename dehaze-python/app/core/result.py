from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from app.core.code import ResultCode
from app.infrastructure.logging import _trace_id_var

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """参数校验错误项"""
    field: str = Field(description="字段名")
    message: str = Field(description="错误消息")
    code: str = Field(description="校验规则码")


class Result(BaseModel, Generic[T]):
    """统一响应结构（与 Java/Go 保持一致，排除 null 字段）"""
    code: str = ResultCode.SUCCESS.code
    msg: str = ResultCode.SUCCESS.msg
    data: Optional[T] = None
    traceId: Optional[str] = Field(default=None, serialization_alias="traceId")
    timestamp: Optional[int] = Field(default=None)
    errors: Optional[list] = Field(default=None)

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)


def _get_trace_id() -> Optional[str]:
    """从 ContextVar 获取当前 trace_id"""
    trace_id = _trace_id_var.get("")
    return trace_id if trace_id else None


def success(data: Any = None, msg: str = ResultCode.SUCCESS.msg) -> Result:
    return Result(code=ResultCode.SUCCESS.code, msg=msg, data=data, traceId=_get_trace_id())


def error(msg: str, code: str = ResultCode.SYSTEM_EXECUTION_ERROR.code) -> Result:
    return Result(code=code, msg=msg, data=None, traceId=_get_trace_id())


def warning(code: ResultCode) -> Result:
    return Result(code=code.code, msg=code.msg, data=None, traceId=_get_trace_id())


def _remove_none(obj: Any) -> Any:
    """递归移除 dict/list 中的 None 值（匹配 Java Jackson NON_NULL 行为）"""
    if isinstance(obj, dict):
        return {k: _remove_none(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [_remove_none(item) for item in obj]
    return obj


def success_response(data: Any = None, msg: str = ResultCode.SUCCESS.msg) -> dict:
    return _remove_none({"code": ResultCode.SUCCESS.code, "msg": msg, "data": data})


def error_response(msg: str, code: str = ResultCode.SYSTEM_EXECUTION_ERROR.code) -> dict:
    return _remove_none({"code": code, "msg": msg, "data": None})
