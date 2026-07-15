from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from app.core.code import ResultCode
from app.infrastructure.logging import _trace_id_var

T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    """统一响应结构（与 Java/Go 保持一致，排除 null 字段）"""
    code: str = ResultCode.SUCCESS.code
    msg: str = ResultCode.SUCCESS.msg
    data: Optional[T] = None
    traceId: Optional[str] = Field(default=None, serialization_alias="traceId")

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
