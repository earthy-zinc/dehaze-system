from typing import Literal

from pydantic import BaseModel, Field

from app.models.schema.common import BasePageQuery


class PromotionForm(BaseModel):
    id: int | None = None
    name: str = Field(..., min_length=1, description="活动名称")
    type: Literal["discount", "new_user", "holiday", "full_reduction"] = Field(
        ..., description="活动类型"
    )
    description: str | None = None
    startTime: str = Field(..., description="活动开始时间")
    endTime: str = Field(..., description="活动结束时间")
    activityRules: dict | None = None
    applicableScope: list[int] | None = None
    newUserOnly: int = Field(default=0, ge=0, le=1, description="是否仅新用户(1:是;0:否)")
    status: int = Field(default=0, ge=0, le=1, description="状态(1:启用;0:禁用)")


class PromotionQuery(BasePageQuery):
    name: str | None = None
    type: Literal["discount", "new_user", "holiday", "full_reduction"] | None = None
    status: int | None = Field(default=None, ge=0, le=1, description="状态(1:启用;0:禁用)")
    startTime: str | None = None
    endTime: str | None = None


class PromotionPackageForm(BaseModel):
    packageIds: list[int] = Field(..., min_length=1, description="关联商品ID列表")
