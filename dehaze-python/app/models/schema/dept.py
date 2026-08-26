"""
部门模块 Schema 模型
"""

from pydantic import BaseModel, Field, field_validator

from app.models.schema.common import validate_no_xss


class DeptForm(BaseModel):
    """部门表单"""

    id: int | None = Field(default=None, description="部门ID")
    parentId: int = Field(..., description="父部门ID")
    name: str = Field(..., min_length=1, max_length=64, description="部门名称")
    sort: int | None = Field(default=0, ge=0, description="排序(数字越小排名越靠前)")
    status: int | None = Field(default=1, ge=0, le=1, description="状态(1-启用；0-禁用)")

    name_no_xss_validator = field_validator("name")(validate_no_xss)


class DeptVO(BaseModel):
    """部门树形VO"""

    id: int = Field(description="部门ID")
    parentId: int = Field(description="父部门ID")
    name: str = Field(description="部门名称")
    sort: int = Field(description="排序")
    status: int = Field(description="状态(1-启用；0-禁用)")
    children: list["DeptVO"] | None = Field(default=None, description="子部门列表")
    createTime: str | None = Field(default=None, description="创建时间")
    updateTime: str | None = Field(default=None, description="修改时间")


class DeptOptionVO(BaseModel):
    """部门下拉选项VO"""

    value: int = Field(description="选项值(部门ID)")
    label: str = Field(description="选项标签(部门名称)")
    children: list["DeptOptionVO"] | None = Field(default=None, description="子选项列表")


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
