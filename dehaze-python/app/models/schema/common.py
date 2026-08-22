"""
公共 Schema 模型 - 可复用的基础模型定义
"""

from typing import Generic, List, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

T = TypeVar("T")


class BasePageQuery(BaseModel):
    """基础分页查询参数"""

    pageNum: int = Field(default=1, ge=1, description="页码")
    pageSize: int = Field(default=10, ge=1, le=100, description="每页记录数")


class PageResult(BaseModel, Generic[T]):
    """分页结果"""

    # 字段名 list 遮蔽内建类型，Pydantic 解析注解时类命名空间含该字段，必须用 typing.List
    list: List[T] = Field(description="数据列表")
    total: int = Field(description="总记录数")


class BatchDeleteForm(BaseModel):
    """批量删除表单（RequestBody JSON）"""

    ids: list[int] = Field(..., min_length=1, description="ID列表")


def validate_no_xss(value: str) -> str:
    """校验字符串不包含 XSS 攻击（HTML 标签或 javascript: 协议）"""
    if value:
        lower_val = value.lower()
        if "javascript:" in lower_val:
            raise ValueError("名称不能包含 javascript: 脚本")
        import re

        if re.search(r"<[a-zA-Z]", value):
            raise ValueError("名称不能包含 HTML 标签")
    return value


class OrmResult(BaseModel):
    """ORM 结果基类：字段用 snake_case 定义（与实体一致），序列化输出 camelCase。

    用法：
    - Result 模型继承 OrmResult，字段名与 ORM 实体一致（snake_case）
    - Service 层用 ModelClass.model_validate(orm_entity) 构建
    - FastAPI response_model 序列化时自动输出 camelCase
    """

    model_config = ConfigDict(
        from_attributes=True,
        alias_generator=to_camel,
        populate_by_name=True,
        protected_namespaces=(),
    )
