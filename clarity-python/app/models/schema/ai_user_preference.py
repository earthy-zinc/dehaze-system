"""用户 AI 偏好 Schema 模型"""

from pydantic import Field

from app.models.schema.common import OrmResult


class AiUserPreferenceUpdate(OrmResult):
    """偏好默认对话模型表单（PUT 全量替换：字段缺省与 null 均=清除偏好，三端一致）"""

    default_model: str | None = Field(
        default=None,
        max_length=64,
        description="偏好默认对话模型(model_id;null/缺省=清除偏好,回落平台默认模型)",
    )


class AiUserPreferenceResult(OrmResult):
    """用户偏好默认对话模型"""

    default_model: str | None = Field(
        default=None, description="偏好默认对话模型(model_id;null=未设置,跟随平台默认)"
    )
