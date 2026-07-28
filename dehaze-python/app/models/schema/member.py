from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.schema.common import BasePageQuery


class MemberPageQuery(BasePageQuery):
    keywords: Optional[str] = None
    levelCode: Optional[str] = None
    status: Optional[int] = Field(default=None, ge=0, le=1)
    expireTimeStart: Optional[str] = None
    expireTimeEnd: Optional[str] = None
    growthMin: Optional[int] = None
    growthMax: Optional[int] = None


class GrowthLogQuery(BasePageQuery):
    changeType: Optional[str] = None
    startTime: Optional[str] = None
    endTime: Optional[str] = None


class MemberLevelAdjustForm(BaseModel):
    levelCode: str = Field(..., min_length=1, description="目标等级")
    expireTime: Optional[str] = None
    reason: str = Field(default="", description="调整原因")


class MemberGrowthAdjustForm(BaseModel):
    changeValue: int = Field(..., description="变动值(正数增加/负数扣减)")
    reason: str = Field(default="", description="调整原因")


class MemberStatusForm(BaseModel):
    status: int = Field(..., ge=0, le=1, description="状态(1:正常;0:冻结)")
    reason: Optional[str] = None


class BenefitForm(BaseModel):
    levelName: Optional[str] = None
    growthMin: Optional[int] = None
    growthMax: Optional[int] = None
    monthlyDehazeQuota: Optional[int] = None
    monthlyEvaluateQuota: Optional[int] = None
    historyRetention: Optional[int] = None
    batchLimit: Optional[int] = None
    priority: Optional[int] = Field(default=None, ge=1, le=4)
    advancedParams: Optional[int] = Field(default=None, ge=0, le=1)
    hdExport: Optional[int] = Field(default=None, ge=0, le=1)
    reportExport: Optional[int] = Field(default=None, ge=0, le=1)
    batchDownload: Optional[int] = Field(default=None, ge=0, le=1)
    sort: Optional[int] = None
    status: Optional[int] = Field(default=None, ge=0, le=1)


class BenefitVO(BaseModel):
    levelCode: str
    levelName: str
    growthMin: int
    growthMax: int
    monthlyDehazeQuota: int
    monthlyEvaluateQuota: int
    historyRetention: int
    batchLimit: int
    priority: int
    advancedParams: int
    hdExport: int
    reportExport: int
    batchDownload: int
    sort: int
    status: int


class MemberProfileVO(BaseModel):
    userId: int
    username: str
    nickname: Optional[str] = None
    avatar: Optional[str] = None
    levelCode: str
    levelName: str
    growthValue: int
    nextLevelGrowth: Optional[int] = None
    progressPercent: int
    expireTime: Optional[str] = None
    monthlyDehazeQuota: int
    monthlyDehazeUsed: int
    monthlyEvaluateQuota: int
    monthlyEvaluateUsed: int
    benefits: BenefitVO
    status: int


class MemberPageVO(BaseModel):
    userId: int
    username: str
    nickname: Optional[str] = None
    levelCode: str
    levelName: str
    growthValue: int
    monthlyUsed: int
    expireTime: Optional[str] = None
    status: int
    becomeMemberTime: Optional[str] = None


class GrowthLogVO(BaseModel):
    id: int
    changeType: str
    changeValue: int
    balance: int
    relatedId: Optional[str] = None
    reason: Optional[str] = None
    operatorId: Optional[int] = None
    createTime: str


class SignInResultVO(BaseModel):
    signDate: str
    continuousDays: int
    growthValue: int
    bonusGrowth: int


class SignInCalendarVO(BaseModel):
    signDates: list[str]
    continuousDays: int
    totalDays: int


class MemberDetailVO(MemberProfileVO):
    levelSource: str
    totalConsumption: int
    becomeMemberTime: Optional[str] = None
    frozenReason: Optional[str] = None
    frozenTime: Optional[str] = None
    quotaResetMonth: Optional[int] = None
