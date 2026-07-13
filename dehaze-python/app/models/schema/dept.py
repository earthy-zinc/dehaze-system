"""
部门模块 Schema 模型
"""
from typing import List, Optional

from app.models.schema.common import validate_no_xss
from pydantic import BaseModel, Field, field_validator

# ==================== 查询参数模型 ====================


class DeptQuery(BaseModel):
    """部门查询参数"""
    keywords: Optional[str] = Field(default=None, description="关键字(部门名称)")
    status: Optional[int] = Field(
        default=None, ge=0, le=1, description="状态(1-启用；0-禁用)")


# ==================== 路径参数模型 ====================

class DeptIdPath(BaseModel):
    """部门ID路径参数"""
    dept_id: int = Field(..., description="部门ID")


class DeptIdsPath(BaseModel):
    """批量删除路径参数"""
    ids: str = Field(..., description="部门ID，多个以英文逗号(,)分隔")


# ==================== 请求体模型 ====================

class DeptForm(BaseModel):
    """部门表单"""
    id: Optional[int] = Field(default=None, description="部门ID")
    parentId: int = Field(..., description="父部门ID")
    name: str = Field(..., min_length=1, max_length=64, description="部门名称")
    sort: Optional[int] = Field(default=0, ge=0, description="排序(数字越小排名越靠前)")
    status: Optional[int] = Field(
        default=1, ge=0, le=1, description="状态(1-启用；0-禁用)")

    @field_validator('name')
    @classmethod
    def validate_name_no_xss(cls, v):
        return validate_no_xss(v)


# ==================== 响应模型 ====================

class DeptVO(BaseModel):
    """部门树形VO"""
    id: int = Field(description="部门ID")
    parentId: int = Field(description="父部门ID")
    name: str = Field(description="部门名称")
    sort: int = Field(description="排序")
    status: int = Field(description="状态(1-启用；0-禁用)")
    children: Optional[List["DeptVO"]] = Field(
        default=None, description="子部门列表")
    createTime: Optional[str] = Field(default=None, description="创建时间")
    updateTime: Optional[str] = Field(default=None, description="修改时间")


class DeptOptionVO(BaseModel):
    """部门下拉选项VO"""
    value: int = Field(description="选项值(部门ID)")
    label: str = Field(description="选项标签(部门名称)")
    children: Optional[List["DeptOptionVO"]] = Field(
        default=None, description="子选项列表")


class DeptFormVO(BaseModel):
    """部门表单VO"""
    id: int = Field(description="部门ID")
    parentId: int = Field(description="父部门ID")
    name: str = Field(description="部门名称")
    sort: int = Field(description="排序")
    status: int = Field(description="状态(1-启用；0-禁用)")


# 树形结构自引用，需要调用 model_rebuild() 完成模型构建
DeptVO.model_rebuild()
DeptOptionVO.model_rebuild()
