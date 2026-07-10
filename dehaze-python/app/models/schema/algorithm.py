"""
算法模块 Schema 模型
"""
from typing import List, Optional

from pydantic import BaseModel, Field

# ==================== 查询参数模型 ====================


class AlgorithmQuery(BaseModel):
    """算法查询参数"""
    keywords: Optional[str] = Field(default=None, description="关键词(算法名称)")


class AlgorithmIdsQuery(BaseModel):
    """批量删除查询参数"""
    ids: str = Field(..., description="算法ID，多个以英文逗号(,)分隔")


# ==================== 路径参数模型 ====================

class AlgorithmIdPath(BaseModel):
    """算法ID路径参数"""
    algorithm_id: int = Field(..., description="算法ID")


# ==================== 请求体模型 ====================

class AlgorithmForm(BaseModel):
    """算法表单"""
    id: Optional[int] = Field(default=None, description="算法ID")
    parentId: int = Field(
        default=0, ge=0, alias="parentId", description="父级ID")
    type: Optional[str] = Field(default="", description="算法类型")
    name: str = Field(..., min_length=1, max_length=100, description="算法名称")
    path: Optional[str] = Field(default="", description="模型路径")
    importPath: Optional[str] = Field(
        default="", alias="importPath", description="导入路径")
    description: Optional[str] = Field(default="", description="算法描述")
    status: int = Field(default=1, ge=0, le=1, description="状态(1-启用；0-停用)")

    model_config = {"populate_by_name": True}


# ==================== 响应模型 ====================

class AlgorithmVO(BaseModel):
    """算法VO"""
    id: int = Field(description="算法ID")
    parent_id: int = Field(description="父级ID")
    type: Optional[str] = Field(default=None, description="算法类型")
    name: str = Field(description="算法名称")
    path: Optional[str] = Field(default=None, description="模型路径")
    size: Optional[str] = Field(default=None, description="模型大小")
    img: Optional[str] = Field(default=None, description="示例图片")
    params: Optional[str] = Field(default=None, description="参数量")
    flops: Optional[str] = Field(default=None, description="计算量")
    import_path: Optional[str] = Field(default=None, description="导入路径")
    description: Optional[str] = Field(default=None, description="算法描述")
    status: int = Field(description="状态(1-启用；0-停用)")
    create_time: Optional[str] = Field(default=None, description="创建时间")
    update_time: Optional[str] = Field(default=None, description="更新时间")
    children: Optional[List["AlgorithmVO"]] = Field(
        default=None, description="子算法列表")


class AlgorithmOptionVO(BaseModel):
    """算法下拉选项VO"""
    value: int = Field(description="选项值(算法ID)")
    label: str = Field(description="选项标签(算法名称)")
    children: Optional[List["AlgorithmOptionVO"]] = Field(
        default=None, description="子选项列表")


class AlgorithmIdVO(BaseModel):
    """算法ID响应VO"""
    id: int = Field(description="算法ID")


class AlgorithmDeleteResultVO(BaseModel):
    """算法删除结果VO"""
    count: int = Field(description="删除数量")


# 解决自引用
AlgorithmVO.model_rebuild()
AlgorithmOptionVO.model_rebuild()
