"""定时调度 Schema 模型（F-M08-009）

请求/响应字段采用 snake_case 定义、序列化输出 camelCase（继承 OrmResult），
与三端后端 API 契约（API接口.md §2.8.4）对齐。
"""

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.models.schema.common import BasePageQuery, OrmResult

INPUT_TYPES = ("fixed", "dynamic")
OUTPUT_TYPES = ("message", "callback")


def _validate_typed_config(
    value: dict[str, Any] | None, allowed: tuple[str, ...], field: str
) -> dict[str, Any] | None:
    """校验 JSON 配置的 type 枚举合法性，不深度校验内部结构。"""
    if value is None:
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 JSON 对象")
    cfg_type = value.get("type")
    if cfg_type not in allowed:
        raise ValueError(f"{field}.type 取值非法，应为 {'/'.join(allowed)}")
    return value


class ScheduleCreate(OrmResult):
    """创建定时任务请求体"""

    name: str = Field(..., min_length=1, max_length=128, description="任务名称")
    cron: str = Field(
        ..., min_length=1, max_length=64, description='Cron触发规则(5位Cron表达式,如"0 9 * * *")'
    )
    timezone: str = Field(
        default="Asia/Shanghai", max_length=64, description="任务时区(默认Asia/Shanghai)"
    )
    input: dict[str, Any] | None = Field(
        default=None, description="输入来源JSON({type:fixed|dynamic,...})"
    )
    output: dict[str, Any] | None = Field(
        default=None, description="输出目标JSON({type:message|callback,...})"
    )

    @field_validator("input")
    @classmethod
    def _check_input(cls, v):
        return _validate_typed_config(v, INPUT_TYPES, "input")

    @field_validator("output")
    @classmethod
    def _check_output(cls, v):
        return _validate_typed_config(v, OUTPUT_TYPES, "output")


class ScheduleUpdate(OrmResult):
    """更新定时任务请求体"""

    name: str | None = Field(default=None, min_length=1, max_length=128, description="任务名称")
    cron: str | None = Field(default=None, min_length=1, max_length=64, description="Cron触发规则")
    timezone: str | None = Field(default=None, max_length=64, description="任务时区")
    input: dict[str, Any] | None = Field(default=None, description="输入来源JSON")
    output: dict[str, Any] | None = Field(default=None, description="输出目标JSON")
    enabled: int | None = Field(default=None, ge=0, le=1, description="用户启停(1:启用;0:停用)")

    @field_validator("input")
    @classmethod
    def _check_input(cls, v):
        return _validate_typed_config(v, INPUT_TYPES, "input")

    @field_validator("output")
    @classmethod
    def _check_output(cls, v):
        return _validate_typed_config(v, OUTPUT_TYPES, "output")


class ScheduleStatusForm(OrmResult):
    """启停任务请求体"""

    enabled: int = Field(..., ge=0, le=1, description="目标启停状态(1:启用;0:停用)")


class ScheduleDetail(OrmResult):
    """定时任务详情"""

    id: int = Field(description="主键")
    userId: int = Field(description="归属用户ID")
    name: str = Field(description="任务名称")
    cron: str = Field(description="Cron触发规则")
    timezone: str = Field(description="任务时区")
    input: dict[str, Any] | None = Field(default=None, description="输入来源JSON")
    output: dict[str, Any] | None = Field(default=None, description="输出目标JSON")
    enabled: int = Field(description="用户启停(1:启用;0:停用)")
    status: int = Field(description="任务状态(1:正常;2:熔断停用)")
    circuitStreak: int = Field(description="连续失败计数")
    nextTriggerTime: datetime | None = Field(default=None, description="下次触发时间")
    createTime: datetime | None = Field(default=None, description="创建时间")


class RunSummary(OrmResult):
    """最近一次执行摘要（列表聚合展示）"""

    status: int = Field(description="执行结果(0:执行中;1:成功;2:失败;3:跳过)")
    skipReason: str | None = Field(default=None, description="跳过原因")
    credits: float | None = Field(default=None, description="消耗积分")
    durationMs: int | None = Field(default=None, description="耗时(毫秒)")
    errorMsg: str | None = Field(default=None, description="失败原因")
    conversationId: int | None = Field(default=None, description="关联会话ID")
    createTime: datetime | None = Field(default=None, description="执行时间")


class ScheduleListItem(ScheduleDetail):
    """定时任务列表项（含最近执行摘要，camelCase 对齐 API 契约）"""

    lastRun: RunSummary | None = Field(default=None, description="最近一次执行摘要")


class RunHistoryItem(OrmResult):
    """执行历史项"""

    id: int = Field(description="主键")
    scheduleId: int = Field(description="关联定时任务ID")
    status: int = Field(description="执行结果(0:执行中;1:成功;2:失败;3:跳过)")
    skipReason: str | None = Field(default=None, description="跳过原因")
    credits: float | None = Field(default=None, description="消耗积分")
    durationMs: int | None = Field(default=None, description="耗时(毫秒)")
    errorMsg: str | None = Field(default=None, description="失败原因")
    conversationId: int | None = Field(default=None, description="关联会话ID")
    requestId: str | None = Field(default=None, description="调用链路ID")
    windowStart: datetime | None = Field(default=None, description="触发窗口")
    createTime: datetime | None = Field(default=None, description="执行时间")


class NextTimesPreview(OrmResult):
    """Cron 解释与下次执行时间预览"""

    description: str = Field(description="Cron 的人类可读描述")
    nextTimes: list[datetime] = Field(default_factory=list, description="接下来 N 次触发时间(ISO)")


class SchedulePageQuery(BasePageQuery):
    """定时任务列表查询参数"""

    keyword: str | None = Field(default=None, description="关键字(按名称模糊搜索)")
