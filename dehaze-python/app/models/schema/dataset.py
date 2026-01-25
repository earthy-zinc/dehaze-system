"""
数据集模块 Schema 模型
用于 flask-openapi3 的 Pydantic 类型注解
"""
from typing import Optional, List
from pydantic import BaseModel, Field

from app.models.schema.common import BasePageQuery


# ==================== 查询参数模型 ====================

class DatasetQuery(BaseModel):
    """数据集列表查询参数"""
    keywords: Optional[str] = Field(default=None, description="关键词(数据集名称)")


class DatasetImagePageQuery(BasePageQuery):
    """数据集图片分页查询参数"""
    pass


class DatasetItemPageQuery(BasePageQuery):
    """数据集项分页查询参数"""
    pass


# ==================== 路径参数模型 ====================

class DatasetIdPath(BaseModel):
    """数据集ID路径参数"""
    dataset_id: int = Field(..., description="数据集ID")


class DatasetIdsQuery(BaseModel):
    """批量删除数据集查询参数"""
    ids: str = Field(..., description="数据集ID列表，多个以英文逗号(,)分隔")


class DatasetItemIdPath(BaseModel):
    """数据项ID路径参数"""
    dataset_item_id: int = Field(..., description="数据项ID")


# ==================== 请求体模型 ====================

class DatasetAddForm(BaseModel):
    """数据集新增表单"""
    parentId: int = Field(default=0, description="父数据集ID")
    name: str = Field(..., min_length=1, max_length=100, description="数据集名称")
    type: Optional[str] = Field(default='', max_length=50, description="数据集类型")
    description: Optional[str] = Field(default='', max_length=500, description="数据集描述")
    path: Optional[str] = Field(default='', max_length=255, description="存储位置")
    status: int = Field(default=1, ge=0, le=1, description="状态(1:启用；0:禁用)")


class DatasetUpdateForm(BaseModel):
    """数据集更新表单"""
    parentId: Optional[int] = Field(default=None, description="父数据集ID")
    name: Optional[str] = Field(default=None, min_length=1, max_length=100, description="数据集名称")
    type: Optional[str] = Field(default=None, max_length=50, description="数据集类型")
    description: Optional[str] = Field(default=None, max_length=500, description="数据集描述")
    path: Optional[str] = Field(default=None, max_length=255, description="存储位置")
    status: Optional[int] = Field(default=None, ge=0, le=1, description="状态(1:启用；0:禁用)")


class DatasetItemCreateForm(BaseModel):
    """数据项创建表单"""
    datasetId: int = Field(..., description="所属数据集ID")
    name: Optional[str] = Field(default=None, max_length=200, description="数据项名称")


class DatasetItemUpdateForm(BaseModel):
    """数据项更新表单"""
    id: int = Field(..., description="数据项ID")
    name: str = Field(..., min_length=1, max_length=200, description="数据项名称")


class DatasetItemDeleteForm(BaseModel):
    """数据项删除表单"""
    datasetItemId: int = Field(..., description="数据项ID")


# ==================== 响应模型 ====================

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
    children: Optional[List['DatasetVO']] = Field(default=None, description="子数据集列表")


class DatasetItemVO(BaseModel):
    """数据集项VO"""
    id: int = Field(description="数据项ID")
    datasetId: int = Field(description="所属数据集ID")
    name: Optional[str] = Field(default=None, description="数据项名称")
    createTime: Optional[str] = Field(default=None, description="创建时间")
    updateTime: Optional[str] = Field(default=None, description="更新时间")


class DatasetOptionVO(BaseModel):
    """数据集下拉选项VO"""
    value: int = Field(description="数据集ID")
    label: str = Field(description="数据集名称")
