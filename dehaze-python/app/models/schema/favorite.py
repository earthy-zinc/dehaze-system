"""
收藏管理模块 Pydantic Schema

Form: 前端提交表单
VO: 返回视图对象
Query: 分页查询参数
"""

from typing import Optional

from pydantic import BaseModel, Field

from app.models.schema.common import BasePageQuery


class FavoriteCreateForm(BaseModel):
    targetType: str = Field(..., description="收藏对象类型(algorithm/result/dataset/image/preset)")
    targetId: int = Field(..., description="收藏对象ID")


class FavoritePageQuery(BasePageQuery):
    targetType: Optional[str] = Field(default=None, description="按类型筛选")
    keywords: Optional[str] = Field(default=None, description="按对象名称关键词搜索")
    sortBy: Optional[str] = Field(default="create_time", description="排序字段(create_time/rating/usage_frequency)")
    sortOrder: Optional[str] = Field(default="desc", description="排序方向(asc/desc)")


class FavoriteVO(BaseModel):
    id: int = Field(description="收藏记录ID")
    userId: int = Field(description="用户ID")
    targetType: str = Field(description="收藏对象类型")
    targetId: int = Field(description="收藏对象ID")
    targetName: Optional[str] = Field(default=None, description="收藏对象名称（关联查询）")
    targetSummary: Optional[str] = Field(default=None, description="收藏对象摘要信息")
    targetThumbnail: Optional[str] = Field(default=None, description="缩略图URL")
    isInvalid: bool = Field(default=False, description="是否已失效（对象被删除）")
    createTime: Optional[str] = Field(default=None, description="收藏时间")


class FavoriteStatusVO(BaseModel):
    targetType: str = Field(description="收藏对象类型")
    targetId: int = Field(description="收藏对象ID")
    favorited: bool = Field(description="是否已收藏")


class FavoriteCountVO(BaseModel):
    targetType: str = Field(description="收藏对象类型")
    count: int = Field(description="收藏数量")
