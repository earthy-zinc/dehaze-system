from pydantic import BaseModel, Field


class MemberLevelAdjustForm(BaseModel):
    levelCode: str = Field(..., min_length=1, description="目标等级")
    expireTime: str | None = None
    reason: str = Field(default="", description="调整原因")


class MemberGrowthAdjustForm(BaseModel):
    changeValue: int = Field(..., description="变动值(正数增加/负数扣减)")
    reason: str = Field(default="", description="调整原因")


class MemberStatusForm(BaseModel):
    status: int = Field(..., ge=0, le=1, description="状态(1:正常;0:冻结)")
    reason: str | None = None


class BenefitForm(BaseModel):
    levelName: str | None = None
    growthMin: int | None = Field(default=None, ge=0)
    growthMax: int | None = Field(default=None, ge=0)
    monthlyDehazeQuota: int | None = Field(default=None, ge=0)
    monthlyEvaluateQuota: int | None = Field(default=None, ge=0)
    historyRetention: int | None = Field(default=None, ge=0)
    batchLimit: int | None = Field(default=None, ge=0)
    priority: int | None = Field(default=None, ge=1, le=4)
    advancedParams: int | None = Field(default=None, ge=0, le=1)
    hdExport: int | None = Field(default=None, ge=0, le=1)
    reportExport: int | None = Field(default=None, ge=0, le=1)
    batchDownload: int | None = Field(default=None, ge=0, le=1)
    sort: int | None = Field(default=None, ge=0)
    status: int | None = Field(default=None, ge=0, le=1)

