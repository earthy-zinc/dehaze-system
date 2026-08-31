"""
语音引擎管理 Schema 模型（管理端：Provider / Key / Model）
"""

from datetime import datetime
from typing import Any

from pydantic import Field

from app.models.schema.common import OrmResult


class VoiceProviderCreate(OrmResult):
    provider_code: str = Field(..., min_length=1, max_length=32, description="引擎编码(删除后不可复用)")
    engine_type: str = Field(..., min_length=1, max_length=16, description="能力类型(asr/tts)")
    display_name: str = Field(..., min_length=1, max_length=128, description="显示名称")
    api_base_url: str | None = Field(
        default=None, max_length=512, description="引擎API基础地址(local为空)"
    )
    auth_type: str = Field(default="bearer", max_length=32, description="认证方式(bearer/x-api-key/custom)")
    default_headers: dict[str, Any] | None = Field(default=None, description="默认请求头(JSON)")
    is_default: int = Field(default=0, description="该engine_type下默认引擎(0/1)")
    sort_order: int = Field(default=0, description="排序序号")
    health_check_enabled: int = Field(default=1, description="健康检查开关(1:开启;0:关闭)")
    remark: str | None = Field(default=None, max_length=512, description="运维备注")
    status: int = Field(default=1, description="状态(1:启用;0:禁用)")


class VoiceProviderUpdate(OrmResult):
    display_name: str | None = Field(default=None, min_length=1, max_length=128, description="显示名称")
    api_base_url: str | None = Field(default=None, max_length=512, description="引擎API基础地址")
    auth_type: str | None = Field(default=None, max_length=32, description="认证方式")
    default_headers: dict[str, Any] | None = Field(default=None, description="默认请求头(JSON)")
    is_default: int | None = Field(default=None, description="该engine_type下默认引擎(0/1)")
    sort_order: int | None = Field(default=None, description="排序序号")
    health_check_enabled: int | None = Field(default=None, description="健康检查开关(1:开启;0:关闭)")
    remark: str | None = Field(default=None, max_length=512, description="运维备注")
    status: int | None = Field(default=None, description="状态(1:启用;0:禁用)")


class VoiceProviderResult(OrmResult):
    id: int = Field(description="主键")
    provider_code: str = Field(description="引擎编码")
    engine_type: str = Field(description="能力类型(asr/tts)")
    display_name: str = Field(description="显示名称")
    api_base_url: str | None = Field(default=None, description="引擎API基础地址")
    auth_type: str = Field(description="认证方式")
    default_headers: dict[str, Any] | None = Field(default=None, description="默认请求头(JSON)")
    is_default: int = Field(description="该engine_type下默认引擎(0/1)")
    sort_order: int = Field(description="排序序号")
    health_check_enabled: int = Field(description="健康检查开关(1:开启;0:关闭)")
    remark: str | None = Field(default=None, description="运维备注")
    status: int = Field(description="状态(1:启用;0:禁用)")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class VoiceProviderKeyCreate(OrmResult):
    name: str = Field(..., min_length=1, max_length=128, description="Key名称")
    key: str = Field(..., min_length=1, description="Key明文(service层加密后不存储此字段)")
    priority: int = Field(default=0, description="优先级(数字越小越优先)")
    weight: int = Field(default=1, ge=1, description="权重")
    daily_quota: int | None = Field(default=None, ge=1, description="日调用上限")
    rpm_limit: int | None = Field(default=None, ge=0, description="分钟调用频率上限(0=不限)")
    expires_at: datetime | None = Field(default=None, description="过期时间")
    status: int = Field(default=1, description="状态(1:启用;0:禁用)")


class VoiceProviderKeyUpdate(OrmResult):
    name: str | None = Field(default=None, min_length=1, max_length=128, description="Key名称")
    priority: int | None = Field(default=None, description="优先级")
    weight: int | None = Field(default=None, ge=1, description="权重")
    daily_quota: int | None = Field(default=None, ge=1, description="日调用上限")
    rpm_limit: int | None = Field(default=None, ge=0, description="分钟调用频率上限(0=不限)")
    expires_at: datetime | None = Field(default=None, description="过期时间")
    status: int | None = Field(default=None, description="状态(1:启用;0:禁用)")


class VoiceProviderKeyResult(OrmResult):
    id: int = Field(description="主键")
    provider_id: int = Field(description="关联引擎ID")
    name: str = Field(description="Key名称")
    key_prefix: str | None = Field(default=None, description="密钥前缀(展示用)")
    status: int = Field(description="状态(1:启用;0:禁用)")
    priority: int = Field(description="优先级")
    weight: int = Field(description="权重")
    daily_quota: int | None = Field(default=None, description="日调用上限")
    rpm_limit: int | None = Field(default=None, description="分钟调用频率上限(0=不限)")
    expires_at: datetime | None = Field(default=None, description="过期时间")
    last_used_at: datetime | None = Field(default=None, description="最后使用时间")
    last_used_by: int | None = Field(default=None, description="最后使用的用户ID")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class VoiceModelCreate(OrmResult):
    provider_id: int = Field(..., description="关联引擎ID")
    model_id: str = Field(..., min_length=1, max_length=64, description="模型/音色业务编码(删除后不可复用)")
    engine_type: str = Field(..., min_length=1, max_length=16, description="能力类型(asr/tts)")
    model_type: str = Field(..., min_length=1, max_length=16, description="子类型(ASR:stream/offline;TTS:voice)")
    display_name: str = Field(..., min_length=1, max_length=128, description="显示名称")
    params: dict[str, Any] | None = Field(default=None, description="模型参数(JSON)")
    status: int = Field(default=1, description="状态(1:启用;0:禁用)")


class VoiceModelUpdate(OrmResult):
    display_name: str | None = Field(default=None, min_length=1, max_length=128, description="显示名称")
    params: dict[str, Any] | None = Field(default=None, description="模型参数(JSON)")
    status: int | None = Field(default=None, description="状态(1:启用;0:禁用)")


class VoiceModelResult(OrmResult):
    id: int = Field(description="主键")
    provider_id: int = Field(description="关联引擎ID")
    model_id: str = Field(description="模型/音色业务编码")
    engine_type: str = Field(description="能力类型(asr/tts)")
    model_type: str = Field(description="子类型(ASR:stream/offline;TTS:voice)")
    display_name: str = Field(description="显示名称")
    params: dict[str, Any] | None = Field(default=None, description="模型参数(JSON)")
    status: int = Field(description="状态(1:启用;0:禁用)")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")
