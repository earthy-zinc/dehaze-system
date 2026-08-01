"""
算法选择模块 Schema
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class AlgorithmTreeNodeVO(BaseModel):
    """算法选择树节点VO"""
    id: int = Field(description="节点ID")
    name: str = Field(description="节点名称")
    parentId: int = Field(default=0, description="父节点ID")
    type: Optional[str] = Field(default=None, description="节点类型")
    isLeaf: bool = Field(default=False, description="是否叶子节点(算法)")
    children: Optional[List["AlgorithmTreeNodeVO"]] = Field(default=None, description="子节点列表")


class AlgorithmDetailVO(BaseModel):
    """算法详情VO（含样例效果图、评分、使用次数）"""
    id: int = Field(description="算法ID")
    name: str = Field(description="算法名称")
    type: Optional[str] = Field(default=None, description="算法类型")
    description: Optional[str] = Field(default=None, description="算法描述")
    img: Optional[str] = Field(default=None, description="样例效果图")
    params: Optional[str] = Field(default=None, description="参数量")
    flops: Optional[str] = Field(default=None, description="计算量")
    size: Optional[str] = Field(default=None, description="模型大小")
    avgRating: float = Field(default=0, description="平均评分(0-5)")
    usageCount: int = Field(default=0, description="使用次数")


class TestRequest(BaseModel):
    """测试算法效果请求"""
    imageUrl: str = Field(..., alias="imageUrl", description="测试图片URL")

    model_config = {"populate_by_name": True}


class TestResultVO(BaseModel):
    """测试算法效果结果VO"""
    resultUrl: str = Field(description="处理后的图片URL")
    processTime: int = Field(description="处理耗时(毫秒)")


class AlgorithmSearchVO(BaseModel):
    """算法搜索结果VO"""
    id: int = Field(description="算法ID")
    name: str = Field(description="算法名称")
    type: Optional[str] = Field(default=None, description="算法类型")
    description: Optional[str] = Field(default=None, description="算法描述")
    avgRating: float = Field(default=0, description="平均评分")


class CompareRequest(BaseModel):
    """算法对比请求"""
    algorithmIds: List[int] = Field(..., min_length=1, max_length=3, alias="algorithmIds", description="算法ID列表(最多3个)")

    model_config = {"populate_by_name": True}


class AlgorithmCompareVO(BaseModel):
    """算法对比结果VO"""
    algorithmId: int = Field(description="算法ID")
    algorithmName: str = Field(description="算法名称")
    type: Optional[str] = Field(default=None, description="算法类型")
    params: Optional[str] = Field(default=None, description="参数量")
    flops: Optional[str] = Field(default=None, description="计算量")
    description: Optional[str] = Field(default=None, description="算法描述")
    avgRating: float = Field(default=0, description="平均评分")
    usageCount: int = Field(default=0, description="使用次数")
