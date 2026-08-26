"""
推荐管理模块 Pydantic Schema
"""

from pydantic import BaseModel, Field


class ColorDistribution(BaseModel):
    temperature: float = Field(..., description="色温(K)")
    saturation: float = Field(..., description="饱和度(0-1)")


class ImageFeatureAnalysisVO(BaseModel):
    imageMd5: str | None = Field(default=None, description="图像MD5（用于后续查询推荐）")
    hazeLevel: str = Field(..., description="雾霾浓度(light/moderate/heavy)")
    hazeConfidence: float = Field(..., description="雾霾浓度置信度(0-1)")
    sceneType: str = Field(
        ..., description="场景类型(urban/landscape/building/night/backlight/indoor)"
    )
    sceneConfidence: float = Field(..., description="场景类型置信度(0-1)")
    lighting: str = Field(..., description="光照条件(bright/normal/dark/veryDark/backlight)")
    complexity: float = Field(..., description="图像复杂度(0-1)")
    colorDistribution: ColorDistribution = Field(..., description="颜色分布")
    resolution: str = Field(..., description="分辨率(sd/hd/uhd)")
    noiseLevel: str = Field(..., description="噪声水平(low/medium/high)")


class AnalyzeForm(BaseModel):
    imageId: int | None = Field(default=None, description="图片ID(imageId/imageUrl二选一)")
    imageUrl: str | None = Field(default=None, description="图片URL(imageId/imageUrl二选一)")


class RecommendedAlgorithmVO(BaseModel):
    recommendationId: int | None = Field(default=None, description="推荐记录ID（用于提交反馈）")
    algorithmId: int = Field(..., description="算法ID")
    algorithmName: str = Field(..., description="算法名称")
    matchScore: int = Field(..., description="匹配度(0-100)")
    reason: str = Field(..., description="推荐理由")
    rating: float | None = Field(default=None, description="算法评分(0-5)，待真实数据采集后填充")
    estimatedTime: int | None = Field(
        default=None, description="预估耗时(ms)，待真实数据采集后填充"
    )
    effectDescription: str = Field(..., description="预期效果描述")


class RecommendationFeedbackForm(BaseModel):
    recommendationId: int = Field(..., description="推荐记录ID")
    useful: bool = Field(..., description="是否有用")


class RecommendationRuleVO(BaseModel):
    id: int = Field(..., description="规则ID")
    ruleName: str = Field(..., description="规则名称")
    sceneType: str = Field(..., description="场景类型")
    algorithmIds: list[int] = Field(..., description="候选算法ID列表")
    weight: int = Field(..., description="规则权重(0-100)")
    enabled: bool = Field(..., description="是否启用")


class RecommendationRuleForm(BaseModel):
    id: int | None = Field(default=0, description="规则ID(0表示新增)")
    ruleName: str = Field(..., description="规则名称")
    sceneType: str = Field(..., description="场景类型")
    algorithmIds: list[int] = Field(..., description="候选算法ID列表")
    weight: int = Field(..., ge=0, le=100, description="规则权重(0-100)")
    enabled: bool = Field(default=True, description="是否启用")


class TrendItem(BaseModel):
    date: str = Field(..., description="日期")
    adoptionRate: float = Field(..., description="采纳率")


class RecommendationReportVO(BaseModel):
    totalRecommendations: int = Field(..., description="总推荐次数")
    adoptionRate: float = Field(..., description="采纳率")
    satisfactionRate: float = Field(..., description="满意度")
    coverageRate: float = Field(..., description="覆盖率")
    coldStartSuccessRate: float = Field(..., description="冷启动成功率")
    trend: list[TrendItem] = Field(..., description="趋势数据")


class IdVO(BaseModel):
    id: int = Field(..., description="ID")
