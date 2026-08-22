"""
字典模块 Schema 模型
"""

from pydantic import BaseModel, Field, field_validator

from app.models.schema.common import validate_no_xss

# ==================== 查询参数模型 ====================


# ==================== 路径参数模型 ====================


# ==================== 请求体模型 ====================


class DictForm(BaseModel):
    """字典表单"""

    id: int | None = Field(default=None, description="字典ID")
    typeCode: str = Field(..., min_length=1, max_length=50, description="字典类型编码")
    name: str = Field(..., min_length=1, max_length=50, description="字典项名称")
    value: str = Field(..., min_length=1, max_length=50, description="字典项值")
    status: int = Field(default=1, ge=0, le=1, description="状态(1-正常；0-禁用)")
    sort: int = Field(default=1, ge=0, description="排序（默认1）")
    defaulted: int = Field(default=0, ge=0, le=1, description="是否默认(1-是；0-否)")
    remark: str | None = Field(default=None, max_length=255, description="备注")

    name_value_typecode_no_xss_validator = field_validator("name", "value", "typeCode")(
        validate_no_xss
    )


class DictTypeForm(BaseModel):
    """字典类型表单"""

    id: int | None = Field(default=None, description="字典类型ID")
    name: str = Field(..., min_length=1, max_length=64, description="类型名称")
    code: str = Field(..., min_length=1, max_length=32, description="类型编码")
    status: int = Field(default=1, ge=0, le=1, description="状态(1-正常；0-禁用)")
    remark: str | None = Field(default=None, max_length=255, description="备注")

    name_code_no_xss_validator = field_validator("name", "code")(validate_no_xss)


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
    remark: str | None = Field(default=None, description="备注")
    createTime: str | None = Field(default=None, description="创建时间")


class DictTypePageVO(BaseModel):
    """字典类型分页VO"""

    id: int = Field(description="字典类型ID")
    name: str = Field(description="类型名称")
    code: str = Field(description="类型编码")
    status: int = Field(description="状态(1-正常；0-禁用)")
    remark: str | None = Field(default=None, description="备注")
    createTime: str | None = Field(default=None, description="创建时间")


class DictTypeFormVO(BaseModel):
    """字典类型表单响应VO"""

    id: int = Field(description="字典类型ID")
    name: str = Field(description="类型名称")
    code: str = Field(description="类型编码")
    status: int = Field(description="状态(1-正常；0-禁用)")
    remark: str | None = Field(default=None, description="备注")


class DictFormVO(BaseModel):
    """字典表单响应VO"""

    id: int = Field(description="字典ID")
    typeCode: str = Field(description="字典类型编码")
    name: str = Field(description="字典项名称")
    value: str = Field(description="字典项值")
    status: int = Field(description="状态(1-正常；0-禁用)")
    defaulted: int = Field(default=0, description="是否默认(1-是；0-否)")
    sort: int = Field(description="排序")
    remark: str | None = Field(default=None, description="备注")


class DictOptionVO(BaseModel):
    """字典下拉选项VO"""

    value: str = Field(description="字典项值")
    label: str = Field(description="字典项名称")
