"""
数据集模块 Schema 模型

按 API 接口文档拆分为三组：数据集、数据项、图片文件
"""
from typing import List, Optional

from app.models.schema.common import BasePageQuery, validate_no_xss
from pydantic import BaseModel, Field, field_validator

# ============================================================
# 数据集 (Dataset)
# ============================================================


class DatasetQuery(BaseModel):
    """数据集列表查询参数（对齐 Java DatasetQuery）"""
    keyword: Optional[str] = Field(default=None, description="关键词(数据集名称)")
    type: Optional[str] = Field(default=None, description="数据集类型")
    status: Optional[int] = Field(default=None, description="状态(1:启用；0:禁用)")


class DatasetAddForm(BaseModel):
    """数据集新增表单（对齐 Java DatasetAddForm）"""
    parentId: int = Field(default=0, description="父数据集ID")
    name: str = Field(..., min_length=1, max_length=255, description="数据集名称")
    type: Optional[str] = Field(default='', max_length=50, description="数据集类型")
    description: Optional[str] = Field(
        default='', max_length=500, description="数据集描述")
    path: Optional[str] = Field(default='', max_length=255, description="存储位置")
    status: int = Field(default=1, ge=0, le=1, description="状态(1:启用；0:禁用)")

    @field_validator('name')
    @classmethod
    def validate_name_no_xss(cls, v):
        return validate_no_xss(v)


class DatasetUpdateForm(BaseModel):
    """数据集更新表单（对齐 Java DatasetUpdateForm）"""
    parentId: Optional[int] = Field(default=None, description="父数据集ID")
    name: Optional[str] = Field(
        default=None, min_length=1, max_length=255, description="数据集名称")
    type: Optional[str] = Field(
        default=None, max_length=50, description="数据集类型")
    description: Optional[str] = Field(
        default=None, max_length=500, description="数据集描述")
    path: Optional[str] = Field(
        default=None, max_length=255, description="存储位置")
    status: Optional[int] = Field(
        default=None, ge=0, le=1, description="状态(1:启用；0:禁用)")

    @field_validator('name')
    @classmethod
    def validate_name_no_xss(cls, v):
        return validate_no_xss(v)


class DatasetStatisticsVO(BaseModel):
    """数据集统计信息"""
    itemCount: int = Field(default=0, description="数据项数量")
    fileCount: int = Field(default=0, description="文件总数")
    totalSize: int = Field(default=0, description="总大小(字节)")
    annotatedCount: int = Field(default=0, description="已标注图片数量")
    unannotatedCount: int = Field(default=0, description="未标注图片数量")
    sceneDistribution: Optional[dict] = Field(default=None, description="场景分布")
    hazeDistribution: Optional[dict] = Field(
        default=None, description="雾霾程度分布")
    formatDistribution: Optional[dict] = Field(
        default=None, description="格式分布")


class DatasetVO(BaseModel):
    """数据集VO"""
    id: int = Field(description="数据集ID")
    parentId: int = Field(description="父数据集ID")
    name: str = Field(description="数据集名称")
    type: Optional[str] = Field(default=None, description="数据集类型")
    description: Optional[str] = Field(default=None, description="数据集描述")
    path: Optional[str] = Field(default=None, description="存储位置")
    status: int = Field(description="状态(1:启用；0:禁用)")
    itemCount: Optional[int] = Field(default=0, description="数据项数量")
    fileCount: Optional[int] = Field(default=0, description="文件数量")
    totalSize: Optional[int] = Field(default=0, description="总大小(字节)")
    statistics: Optional[DatasetStatisticsVO] = Field(
        default=None, description="统计信息")
    createTime: Optional[str] = Field(default=None, description="创建时间")
    updateTime: Optional[str] = Field(default=None, description="更新时间")
    children: Optional[List['DatasetVO']] = Field(
        default=None, description="子数据集列表")


class DatasetOptionVO(BaseModel):
    """数据集下拉选项VO"""
    value: int = Field(description="数据集ID")
    label: str = Field(description="数据集名称")
    children: Optional[List['DatasetOptionVO']] = Field(
        default=None, description="子选项")


class DatasetIdVO(BaseModel):
    """数据集ID响应VO"""
    id: int = Field(description="数据集ID")


class DatasetDeleteResultItemVO(BaseModel):
    """数据集删除结果项VO"""
    datasetId: int = Field(description="数据集ID")
    status: str = Field(description="状态(success/failed)")
    message: Optional[str] = Field(default=None, description="失败原因")


class DatasetDeleteResultVO(BaseModel):
    """数据集批量删除结果VO"""
    total: int = Field(description="总数")
    succeeded: int = Field(description="成功数量")
    failed: int = Field(description="失败数量")
    results: List[DatasetDeleteResultItemVO] = Field(
        default_factory=list, description="删除结果详情")


# ============================================================
# 数据项 (DatasetItem)
# ============================================================

class DatasetItemQuery(BasePageQuery):
    """数据项分页查询参数"""
    datasetId: int = Field(..., description="所属数据集ID")
    keywords: Optional[str] = Field(default=None, description="搜索关键词")


class DatasetItemCreateForm(BaseModel):
    """创建空数据项表单"""
    datasetId: int = Field(..., description="所属数据集ID")
    name: Optional[str] = Field(
        default=None, max_length=200, description="数据项名称")


class DatasetItemUpdateForm(BaseModel):
    """数据项更新表单"""
    name: Optional[str] = Field(
        default=None, min_length=1, max_length=200, description="数据项名称")
    sceneType: Optional[str] = Field(default=None, description="场景类型")


class ItemFileVO(BaseModel):
    """数据项文件VO（对齐 Java 丰富响应格式）"""
    id: int = Field(description="文件关联ID")
    itemId: int = Field(description="数据项ID")
    type: str = Field(description="图片类型(clear/hazy/depth/segment)")
    sceneType: Optional[str] = Field(default=None, description="场景类型")
    hazeLevel: Optional[str] = Field(
        default=None, description="雾霾等级(light/medium/heavy)")
    description: Optional[str] = Field(default=None, description="描述")
    url: Optional[str] = Field(default=None, description="文件URL")
    thumbnailUrl: Optional[str] = Field(default=None, description="缩略图URL")
    md5: Optional[str] = Field(default=None, description="文件MD5")
    fileName: Optional[str] = Field(default=None, description="文件名")
    format: Optional[str] = Field(default=None, description="文件格式")
    formattedSize: Optional[str] = Field(default=None, description="文件大小(格式化)")
    datasetId: Optional[int] = Field(default=None, description="数据集ID")
    datasetName: Optional[str] = Field(default=None, description="数据集名称")
    hasPairedImages: Optional[bool] = Field(default=None, description="是否有配对图")
    pairedCount: Optional[int] = Field(default=None, description="配对图片数")
    usageCount: Optional[int] = Field(default=None, description="使用次数")


class DatasetItemVO(BaseModel):
    """数据项VO"""
    id: int = Field(description="数据项ID")
    datasetId: int = Field(description="所属数据集ID")
    name: Optional[str] = Field(default=None, description="数据项名称")
    createTime: Optional[str] = Field(default=None, description="创建时间")
    updateTime: Optional[str] = Field(default=None, description="更新时间")
    files: List[ItemFileVO] = Field(default_factory=list, description="关联文件列表")
    clearImage: Optional[ItemFileVO] = Field(default=None, description="清晰图信息")
    hazyImages: List[ItemFileVO] = Field(default_factory=list, description="有雾图列表")


class DatasetItemPageVO(BaseModel):
    """数据项分页VO"""
    list: List[DatasetItemVO] = Field(
        default_factory=list, description="数据项列表")
    total: int = Field(description="总数")
    pageNum: int = Field(description="页码")
    pageSize: int = Field(description="每页数量")


class DatasetItemIdVO(BaseModel):
    """数据项ID响应VO"""
    id: int = Field(description="数据项ID")


class DatasetItemDeleteForm(BaseModel):
    """数据项删除表单（兼容旧接口）"""
    datasetItemId: int = Field(..., description="数据项ID")


# ============================================================
# 图片文件 (ItemFile)
# ============================================================

class ItemFileAddForm(BaseModel):
    """上传数据项图片表单（元数据部分，文件通过 UploadFile 传递）"""
    itemId: int = Field(..., description="所属数据项ID")
    type: str = Field(..., description="图片类型(clear/hazy/depth/segment)")
    sceneType: Optional[str] = Field(
        default=None, max_length=64, description="场景类型")
    hazeLevel: Optional[str] = Field(
        default=None, max_length=32, description="雾霾等级(light/medium/heavy)")
    description: Optional[str] = Field(
        default=None, max_length=255, description="描述")


class ItemFileUpdateForm(BaseModel):
    """修改图片信息表单"""
    type: Optional[str] = Field(
        default=None, description="图片类型(clear/hazy/depth/segment)")
    sceneType: Optional[str] = Field(
        default=None, max_length=64, description="场景类型")
    hazeLevel: Optional[str] = Field(
        default=None, max_length=32, description="雾霾等级")
    description: Optional[str] = Field(
        default=None, max_length=255, description="描述")


# ============================================================
# 数据集项上传（配对上传 + 批量上传）
# ============================================================

class BatchUploadSuccessItemVO(BaseModel):
    """批量上传成功项"""
    id: int = Field(description="数据项ID")
    name: Optional[str] = Field(default=None, description="数据项名称")
    fileCount: int = Field(description="关联文件数量")


class BatchUploadFailedItemVO(BaseModel):
    """批量上传失败项"""
    fileName: str = Field(description="文件名")
    reason: str = Field(description="失败原因")


class BatchUploadResultVO(BaseModel):
    """批量上传结果VO"""
    total: int = Field(description="总文件数")
    succeeded: int = Field(description="成功数量")
    failed: int = Field(description="失败数量")
    successItems: List[BatchUploadSuccessItemVO] = Field(
        default_factory=list, description="成功项列表")
    failedItems: List[BatchUploadFailedItemVO] = Field(
        default_factory=list, description="失败项列表")


class BatchActionFailureDetailVO(BaseModel):
    """批量操作失败详情"""
    identifier: Optional[str] = Field(default=None, description="失败记录标识")
    reason: str = Field(description="失败原因")


class BatchOperationResultVO(BaseModel):
    """批量操作结果VO"""
    successCount: int = Field(description="成功数量")
    failedCount: int = Field(description="失败数量")
    message: str = Field(description="操作消息")
    successIds: Optional[List[int]] = Field(
        default=None, description="成功的ID列表")
    failureDetails: Optional[List[BatchActionFailureDetailVO]] = Field(
        default=None, description="失败详情列表")


# 树形结构自引用，需要调用 model_rebuild() 完成模型构建
DatasetVO.model_rebuild()
DatasetOptionVO.model_rebuild()
