"""错误码跨端契约守卫（免库）。

`ResultCode` 成员名是 python 内部命名，码值与中文文案是**跨端契约**（java `ResultCode`、
go `common.ResultCode` 同码同文案，SDK/前端按码分支）：枚举成员可改名，码值与文案不得漂移。
"""

from app.core.code import ResultCode


def test_a0504_code_and_message_are_frozen():
    """A0504「存在关联数据，无法删除」的码值与文案为三端契约，改名不得动它"""
    assert ResultCode.DATA_REFERENCED.code == "A0504"
    assert ResultCode.DATA_REFERENCED.msg == "存在关联数据，无法删除"


def test_misnamed_bind_constant_is_removed_without_alias():
    """旧名 DATA_BIND_EXISTS（名实不符）已删除，且不得以别名方式保留"""
    assert not hasattr(ResultCode, "DATA_BIND_EXISTS")
