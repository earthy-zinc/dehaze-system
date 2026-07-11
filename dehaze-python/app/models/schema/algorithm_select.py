"""
算法选择模块 Schema
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """智能推荐请求"""
    imageUrl: str = Field(..., alias="imageUrl", description="待去雾图片URL")
    topN: int = Field(default=3, ge=1, le=10, description="推荐数量")

    model_config = {"populate_by_name": True}


class AlgorithmRecommendVO(BaseModel):
    """算法推荐结果VO"""
    algorithmId: int = Field(description="算法ID")
    algorithmName: str = Field(description="算法名称")
    score: float = Field(description="匹配得分(0-100)")
    reason: str = Field(default="", description="推荐理由")
    type: Optional[str] = Field(default=None, description="算法类型")


class FavoriteForm(BaseModel):
    """收藏表单"""
    algorithmId: int = Field(..., alias="algorithmId", description="算法ID")

    model_config = {"populate_by_name": True}


class FavoriteVO(BaseModel):
    """收藏VO"""
    id: int = Field(description="收藏ID")
    userId: int = Field(validation_alias="user_id", serialization_alias="userId", description="用户ID")
    algorithmId: int = Field(validation_alias="algorithm_id", serialization_alias="algorithmId", description="算法ID")
    algorithmName: Optional[str] = Field(default=None, description="算法名称")
    createTime: Optional[str] = Field(default=None, validation_alias="create_time", serialization_alias="createTime", description="收藏时间")

    model_config = {"populate_by_name": True}


class CompareRequest(BaseModel):
    """算法对比请求"""
    algorithmIds: List[int] = Field(..., min_length=2, max_length=4, alias="algorithmIds", description="算法ID列表(2-4个)")
    imageUrl: Optional[str] = Field(default=None, alias="imageUrl", description="待对比的图片URL")

    model_config = {"populate_by_name": True}


class AlgorithmCompareVO(BaseModel):
    """算法对比结果VO"""
    algorithmId: int = Field(description="算法ID")
    algorithmName: str = Field(description="算法名称")
    type: Optional[str] = Field(default=None, description="算法类型")
    params: Optional[str] = Field(default=None, description="参数量")
    flops: Optional[str] = Field(default=None, description="计算量")
    description: Optional[str] = Field(default=None, description="算法描述")
    status: int = Field(description="状态")
    # 对比指标（如有预测结果）
    resultUrl: Optional[str] = Field(default=None, description="去雾结果URL")
    processTime: Optional[int] = Field(default=None, description="处理耗时(毫秒)")
