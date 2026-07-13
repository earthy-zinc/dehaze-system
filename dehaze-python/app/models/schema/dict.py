"""
字典模块 Schema 模型
"""
from typing import Optional

from app.models.schema.common import BasePageQuery, validate_no_xss
from pydantic import BaseModel, Field, field_validator

# ==================== 查询参数模型 ====================


class DictPageQuery(BasePageQuery):
    """字典分页查询参数"""
    keywords: Optional[str] = Field(default=None, description="关键词(字典名称)")
    typeCode: Optional[str] = Field(default=None, description="字典类型编码")


class DictTypePageQuery(BasePageQuery):
    """字典类型分页查询参数"""
    keywords: Optional[str] = Field(default=None, description="关键词(类型名称/类型编码)")


# ==================== 路径参数模型 ====================

class DictIdPath(BaseModel):
    """字典ID路径参数"""
    dict_id: int = Field(..., description="字典ID")


class DictIdsPath(BaseModel):
    """字典批量删除路径参数"""
    dict_ids: str = Field(..., description="字典ID，多个以英文逗号(,)分隔")


class DictTypeIdPath(BaseModel):
    """字典类型ID路径参数"""
    type_id: int = Field(..., description="字典类型ID")


class DictTypeIdsPath(BaseModel):
    """字典类型批量删除路径参数"""
    type_ids: str = Field(..., description="字典类型ID，多个以英文逗号(,)分隔")


class DictTypeCodePath(BaseModel):
    """字典类型编码路径参数"""
    type_code: str = Field(..., description="字典类型编码")


# ==================== 请求体模型 ====================

class DictForm(BaseModel):
    """字典表单"""
    id: Optional[int] = Field(default=None, description="字典ID")
    typeCode: str = Field(..., min_length=1,
                          max_length=50, description="字典类型编码")
    name: str = Field(..., min_length=1, max_length=50, description="字典项名称")
    value: str = Field(..., min_length=1, max_length=50, description="字典项值")
    status: int = Field(default=1, ge=0, le=1, description="状态(1-正常；0-禁用)")
    sort: int = Field(default=0, ge=0, description="排序")
    defaulted: int = Field(default=0, ge=0, le=1, description="是否默认(1-是；0-否)")
    remark: Optional[str] = Field(
        default=None, max_length=255, description="备注")

    @field_validator('name', 'value', 'typeCode')
    @classmethod
    def validate_no_xss(cls, v):
        return validate_no_xss(v)


class DictTypeForm(BaseModel):
    """字典类型表单"""
    id: Optional[int] = Field(default=None, description="字典类型ID")
    name: str = Field(..., min_length=1, max_length=64, description="类型名称")
    code: str = Field(..., min_length=1, max_length=32, description="类型编码")
    status: int = Field(default=1, ge=0, le=1, description="状态(1-正常；0-禁用)")
    remark: Optional[str] = Field(
        default=None, max_length=255, description="备注")

    @field_validator('name', 'code')
    @classmethod
    def validate_no_xss(cls, v):
        return validate_no_xss(v)


# ==================== 响应模型 ====================

class DictPageVO(BaseModel):
    """字典分页VO"""
    id: int = Field(description="字典ID")
    typeCode: str = Field(description="字典类型编码")
    name: str = Field(description="字典项名称")
    value: str = Field(description="字典项值")
    status: int = Field(description="状态(1-正常；0-禁用)")
    defaulted: int = Field(default=0, description="是否默认(1-是；0-否)")
    sort: int = Field(description="排序")
    remark: Optional[str] = Field(default=None, description="备注")
    createTime: Optional[str] = Field(default=None, description="创建时间")


class DictTypePageVO(BaseModel):
    """字典类型分页VO"""
    id: int = Field(description="字典类型ID")
    name: str = Field(description="类型名称")
    code: str = Field(description="类型编码")
    status: int = Field(description="状态(1-正常；0-禁用)")
    remark: Optional[str] = Field(default=None, description="备注")
    createTime: Optional[str] = Field(default=None, description="创建时间")


class DictTypeFormVO(BaseModel):
    """字典类型表单响应VO"""
    id: int = Field(description="字典类型ID")
    name: str = Field(description="类型名称")
    code: str = Field(description="类型编码")
    status: int = Field(description="状态(1-正常；0-禁用)")
    remark: Optional[str] = Field(default=None, description="备注")


class DictFormVO(BaseModel):
    """字典表单响应VO"""
    id: int = Field(description="字典ID")
    typeCode: str = Field(description="字典类型编码")
    name: str = Field(description="字典项名称")
    value: str = Field(description="字典项值")
    status: int = Field(description="状态(1-正常；0-禁用)")
    defaulted: int = Field(default=0, description="是否默认(1-是；0-否)")
    sort: int = Field(description="排序")
    remark: Optional[str] = Field(default=None, description="备注")


class DictOptionVO(BaseModel):
    """字典下拉选项VO"""
    value: str = Field(description="字典项值")
    label: str = Field(description="字典项名称")
