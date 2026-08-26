"""
数据集模块 Schema 模型

按 API 接口文档拆分为三组：数据集、数据项、图片文件
"""

from typing import List

from pydantic import BaseModel, Field, field_validator

from app.models.schema.common import validate_no_xss

# 数据集 (Dataset)


class DatasetAddForm(BaseModel):
    """数据集新增表单（对齐 Java DatasetAddForm）"""

    parentId: int = Field(default=0, description="父数据集ID")
    name: str = Field(..., min_length=1, max_length=255, description="数据集名称")
    type: str | None = Field(default="", max_length=50, description="数据集类型")
    description: str | None = Field(default="", max_length=500, description="数据集描述")
    path: str | None = Field(default="", max_length=255, description="存储位置")
    status: int = Field(default=1, ge=0, le=1, description="状态(1:启用；0:禁用)")

    name_no_xss_validator = field_validator("name")(validate_no_xss)


class DatasetUpdateForm(BaseModel):
    """数据集更新表单（对齐 Java DatasetUpdateForm）"""

    parentId: int | None = Field(default=None, description="父数据集ID")
    name: str | None = Field(default=None, min_length=1, max_length=255, description="数据集名称")
    type: str | None = Field(default=None, max_length=50, description="数据集类型")
    description: str | None = Field(default=None, max_length=500, description="数据集描述")
    path: str | None = Field(default=None, max_length=255, description="存储位置")
    status: int | None = Field(default=None, ge=0, le=1, description="状态(1:启用；0:禁用)")

    name_no_xss_validator = field_validator("name")(validate_no_xss)


# 数据项 (DatasetItem)


class DatasetItemCreateForm(BaseModel):
    """创建空数据项表单"""

    datasetId: int = Field(..., description="所属数据集ID")
    name: str | None = Field(default=None, max_length=200, description="数据项名称")


class DatasetItemUpdateForm(BaseModel):
    """数据项更新表单"""

    name: str | None = Field(default=None, min_length=1, max_length=200, description="数据项名称")
    sceneType: str | None = Field(default=None, description="场景类型")


class ItemFileVO(BaseModel):
    """数据项文件VO（对齐 Java 丰富响应格式）"""

    id: int = Field(description="文件关联ID")
    itemId: int = Field(description="数据项ID")
    type: str = Field(description="图片类型(clear/hazy/depth/segment)")
    sceneType: str | None = Field(default=None, description="场景类型")
    hazeLevel: str | None = Field(default=None, description="雾霾等级(light/medium/heavy)")
    description: str | None = Field(default=None, description="描述")
    url: str | None = Field(default=None, description="文件URL")
    thumbnailUrl: str | None = Field(default=None, description="缩略图URL")
    md5: str | None = Field(default=None, description="文件MD5")
    fileName: str | None = Field(default=None, description="文件名")
    format: str | None = Field(default=None, description="文件格式")
    formattedSize: str | None = Field(default=None, description="文件大小(格式化)")
    datasetId: int | None = Field(default=None, description="数据集ID")
    datasetName: str | None = Field(default=None, description="数据集名称")
    hasPairedImages: bool | None = Field(default=None, description="是否有配对图")
    pairedCount: int | None = Field(default=None, description="配对图片数")
    usageCount: int | None = Field(default=None, description="使用次数")


class DatasetItemVO(BaseModel):
    """数据项VO"""

    id: int = Field(description="数据项ID")
    datasetId: int = Field(description="所属数据集ID")
    name: str | None = Field(default=None, description="数据项名称")
    createTime: str | None = Field(default=None, description="创建时间")
    updateTime: str | None = Field(default=None, description="更新时间")
    files: list[ItemFileVO] = Field(default_factory=list, description="关联文件列表")
    clearImage: ItemFileVO | None = Field(default=None, description="清晰图信息")
    hazyImages: list[ItemFileVO] = Field(default_factory=list, description="有雾图列表")


class DatasetItemPageVO(BaseModel):
    """数据项分页VO"""

    # 字段名 list 遮蔽内建类型，必须用 typing.List
    list: List[DatasetItemVO] = Field(default_factory=list, description="数据项列表")
    total: int = Field(description="总数")
    pageNum: int = Field(description="页码")
    pageSize: int = Field(description="每页数量")


# 图片文件 (ItemFile)


class ItemFileUpdateForm(BaseModel):
    """修改图片信息表单"""

    type: str | None = Field(default=None, description="图片类型(clear/hazy/depth/segment)")
    sceneType: str | None = Field(default=None, max_length=64, description="场景类型")
    hazeLevel: str | None = Field(default=None, max_length=32, description="雾霾等级")
    description: str | None = Field(default=None, max_length=255, description="描述")


# 数据集项上传（配对上传 + 批量上传）


class BatchUploadSuccessItemVO(BaseModel):
    """批量上传成功项"""

    id: int = Field(description="数据项ID")
    name: str | None = Field(default=None, description="数据项名称")
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
    successItems: list[BatchUploadSuccessItemVO] = Field(
        default_factory=list, description="成功项列表"
    )
    failedItems: list[BatchUploadFailedItemVO] = Field(
        default_factory=list, description="失败项列表"
    )


class BatchActionFailureDetailVO(BaseModel):
    """批量操作失败详情"""

    identifier: str | None = Field(default=None, description="失败记录标识")
    reason: str = Field(description="失败原因")


class BatchOperationResultVO(BaseModel):
    """批量操作结果VO"""

    successCount: int = Field(description="成功数量")
    failedCount: int = Field(description="失败数量")
    message: str = Field(description="操作消息")
    successIds: list[int] | None = Field(default=None, description="成功的ID列表")
    failureDetails: list[BatchActionFailureDetailVO] | None = Field(
        default=None, description="失败详情列表"
    )
