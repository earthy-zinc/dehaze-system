"""
收藏管理模块 Pydantic Schema

Form: 前端提交表单
VO: 返回视图对象
Query: 分页查询参数
"""

from pydantic import BaseModel, Field


class FavoriteCreateForm(BaseModel):
    targetType: str = Field(..., description="收藏对象类型(algorithm/result/dataset/image/preset)")
    targetId: int = Field(..., description="收藏对象ID")

