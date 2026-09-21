"""
算法选择模块 Schema
"""

from pydantic import BaseModel, Field, model_validator


class AlgorithmTreeNodeVO(BaseModel):
    """算法选择树节点VO（leaf 字段名对齐 Java AlgorithmSelectNodeVO / 文档 §3.1）"""

    id: int = Field(description="节点ID")
    parentId: int = Field(default=0, description="父节点ID")
    name: str = Field(description="节点名称")
    type: str | None = Field(default=None, description="节点类型")
    leaf: bool = Field(default=True, description="是否叶子节点(算法)")
    children: list["AlgorithmTreeNodeVO"] | None = Field(default=None, description="子节点列表")


class AlgorithmDetailVO(BaseModel):
    """算法详情VO（含样例效果图、评分、使用次数；字段对齐 Java AlgorithmDetailVO）"""

    id: int = Field(description="算法ID")
    name: str = Field(description="算法名称")
    type: str | None = Field(default=None, description="算法类型")
    description: str | None = Field(default=None, description="算法描述")
    img: str | None = Field(default=None, description="样例效果图")
    path: str | None = Field(default=None, description="模型存储路径")
    params: str | None = Field(default=None, description="参数量")
    flops: str | None = Field(default=None, description="计算量")
    size: str | None = Field(default=None, description="模型大小")
    version: str | None = Field(default=None, description="算法版本号")
    status: int | None = Field(default=None, description="算法状态")
    avgRating: float = Field(default=0, description="平均评分(0-5)")
    ratingCount: int = Field(default=0, description="评价总数")
    usageCount: int = Field(default=0, description="使用次数")
    sampleImages: list[str] = Field(
        default_factory=list, description="最近成功预测结果图（最多3张）"
    )


class TestRequest(BaseModel):
    """测试算法效果请求（fileId/imageUrl 二选一，对齐 Java AlgorithmTestForm）"""

    fileId: int | None = Field(default=None, alias="fileId", description="文件ID")
    imageUrl: str | None = Field(default=None, alias="imageUrl", description="测试图片URL")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def require_image_source(self) -> "TestRequest":
        if self.fileId is None and not self.imageUrl:
            raise ValueError("imageUrl 与 fileId 至少提供一个")
        return self


class TestResultVO(BaseModel):
    """测试算法效果结果VO（异步任务契约，对齐 Java PredictionResultVO）"""

    logId: int = Field(description="预测日志ID")
    status: int = Field(description="任务状态：1=处理中,2=已完成,3=失败")
    resultUrl: str | None = Field(
        default=None, description="处理后的图片URL（status=completed 时返回）"
    )
    time: int | None = Field(default=None, description="处理耗时(毫秒)（status=completed 时返回）")


class AlgorithmSearchVO(BaseModel):
    """算法搜索结果VO"""

    id: int = Field(description="算法ID")
    name: str = Field(description="算法名称")
    type: str | None = Field(default=None, description="算法类型")
    description: str | None = Field(default=None, description="算法描述")
    avgRating: float = Field(default=0, description="平均评分")


class CompareRequest(BaseModel):
    """算法对比请求（fileId/imageUrl 二选一）"""

    # T-AS-055：算法对比数量需在 2-3 个之间
    algorithmIds: list[int] = Field(
        ..., min_length=2, max_length=3, alias="algorithmIds", description="算法ID列表(2-3个)"
    )
    fileId: int | None = Field(
        default=None, alias="fileId", description="文件ID（与 imageUrl 二选一）"
    )
    imageUrl: str | None = Field(
        default=None, alias="imageUrl", description="测试图片URL（与 fileId 二选一）"
    )

    model_config = {"populate_by_name": True}


class AlgorithmCompareVO(BaseModel):
    """算法对比结果VO（对齐 Java AlgorithmCompareVO / 文档 §3.3）"""

    algorithmId: int = Field(description="算法ID")
    algorithmName: str = Field(description="算法名称")
    resultUrl: str | None = Field(default=None, description="处理结果URL（服务端预测完成时返回）")
    time: int | None = Field(default=None, description="处理耗时(毫秒)")


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
