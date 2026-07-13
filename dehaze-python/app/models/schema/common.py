"""
公共 Schema 模型 - 可复用的基础模型定义
"""
from typing import TypeVar, Generic, List

from pydantic import BaseModel, Field

T = TypeVar('T')


class BasePageQuery(BaseModel):
    """基础分页查询参数"""
    pageNum: int = Field(default=1, ge=1, description="页码")
    pageSize: int = Field(default=10, ge=1, le=100, description="每页记录数")


class PageResult(BaseModel, Generic[T]):
    """分页结果"""
    list: List[T] = Field(description="数据列表")
    total: int = Field(description="总记录数")


class Option(BaseModel, Generic[T]):
    """下拉选项"""
    value: T = Field(description="选项值")
    label: str = Field(description="选项标签")


class IdsPath(BaseModel):
    """批量操作的 ID 路径参数"""
    ids: str = Field(..., description="ID列表，多个以英文逗号(,)分隔")


class IdPath(BaseModel):
    """单个 ID 路径参数"""
    id: int = Field(..., description="ID")


class BatchDeleteForm(BaseModel):
    """批量删除表单（RequestBody JSON）"""
    ids: List[int] = Field(..., min_length=1, description="ID列表")


def validate_no_xss(value: str) -> str:
    """校验字符串不包含 XSS 攻击（HTML 标签或 javascript: 协议）"""
    if value:
        lower_val = value.lower()
        if 'javascript:' in lower_val:
            raise ValueError('名称不能包含 javascript: 脚本')
        import re
        if re.search(r'<[a-zA-Z]', value):
            raise ValueError('名称不能包含 HTML 标签')
    return value
