"""
算法选择模块 Schema
"""

from pydantic import BaseModel, Field


class AlgorithmTreeNodeVO(BaseModel):
    """算法选择树节点VO"""

    id: int = Field(description="节点ID")
    name: str = Field(description="节点名称")
    parentId: int = Field(default=0, description="父节点ID")
    type: str | None = Field(default=None, description="节点类型")
    isLeaf: bool = Field(default=False, description="是否叶子节点(算法)")
    children: list["AlgorithmTreeNodeVO"] | None = Field(default=None, description="子节点列表")


class AlgorithmDetailVO(BaseModel):
    """算法详情VO（含样例效果图、评分、使用次数）"""

    id: int = Field(description="算法ID")
    name: str = Field(description="算法名称")
    type: str | None = Field(default=None, description="算法类型")
    description: str | None = Field(default=None, description="算法描述")
    img: str | None = Field(default=None, description="样例效果图")
    params: str | None = Field(default=None, description="参数量")
    flops: str | None = Field(default=None, description="计算量")
    size: str | None = Field(default=None, description="模型大小")
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
    type: str | None = Field(default=None, description="算法类型")
    description: str | None = Field(default=None, description="算法描述")
    avgRating: float = Field(default=0, description="平均评分")


class CompareRequest(BaseModel):
    """算法对比请求"""

    # T-AS-055：算法对比数量需在 2-3 个之间（下限 2）
    algorithmIds: list[int] = Field(
        ..., min_length=2, max_length=3, alias="algorithmIds", description="算法ID列表(2-3个)"
    )

    model_config = {"populate_by_name": True}


class AlgorithmCompareVO(BaseModel):
    """算法对比结果VO"""

    algorithmId: int = Field(description="算法ID")
    algorithmName: str = Field(description="算法名称")
    type: str | None = Field(default=None, description="算法类型")
    params: str | None = Field(default=None, description="参数量")
    flops: str | None = Field(default=None, description="计算量")
    description: str | None = Field(default=None, description="算法描述")
    avgRating: float = Field(default=0, description="平均评分")
    usageCount: int = Field(default=0, description="使用次数")


class RecommendRequest(BaseModel):
    """算法推荐匹配请求（F-M03-007）"""

    keyword: str | None = Field(default=None, description="关键词")
    taskType: str | None = Field(default=None, alias="taskType", description="任务类型")
    sampleAlgorithmId: int | None = Field(
        default=None, alias="sampleAlgorithmId", description="样例算法ID"
    )
    topN: int | None = Field(default=None, ge=1, le=10, description="推荐数量(1-10，默认3)")

    model_config = {"populate_by_name": True}


class RecommendItemVO(BaseModel):
    """算法推荐匹配结果项"""

    algorithmId: int = Field(description="算法ID")
    algorithmName: str = Field(description="算法名称")
    matchScore: int = Field(description="匹配度(0-100)")
    reason: str = Field(description="推荐理由")
    estimatedTime: int | None = Field(default=None, description="预估耗时(毫秒)")


class RecommendResultVO(BaseModel):
    """算法推荐匹配结果"""

    total: int = Field(description="结果总数")
    items: list[RecommendItemVO] = Field(default_factory=list, description="推荐列表")
