from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel

from app.core.code import ResultCode

T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    code: str = ResultCode.SUCCESS.code
    msg: str = ResultCode.SUCCESS.msg
    data: Optional[T] = None


def success(data: Any = None, msg: str = ResultCode.SUCCESS.msg) -> Result:
    return Result(code=ResultCode.SUCCESS.code, msg=msg, data=data)


def error(msg: str, code: str =  ResultCode.SYSTEM_EXECUTION_ERROR.code) -> Result:
    return Result(code=code, msg=msg, data=None)


def warning(code: ResultCode) -> Result:
    return Result(code=code.code, msg=code.msg, data=None)


def success_response(data: Any = None, msg: str = ResultCode.SUCCESS.msg) -> dict:
    return {"code": ResultCode.SUCCESS.code, "msg": msg, "data": data}


def error_response(msg: str, code: str = ResultCode.SYSTEM_EXECUTION_ERROR.code) -> dict:
    return {"code": code, "msg": msg, "data": None}
