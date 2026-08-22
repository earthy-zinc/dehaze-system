"""参数预设 Schema"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PresetForm(BaseModel):
    """创建/更新预设请求"""

    name: str = Field(description="预设名称")
    algorithmId: int = Field(description="关联算法ID")
    params: Any | None = Field(default=None, description="参数键值对(JSON或对象)")
    isDefault: int | None = Field(default=0, description="是否默认预设")


class PresetVO(BaseModel):
    """预设视图对象"""

    id: int = Field(description="预设ID")
    name: str = Field(description="预设名称")
    type: str = Field(description="预设类型(system:系统预设;custom:用户自定义)")
    algorithmId: int = Field(description="关联算法ID")
    params: Any | None = Field(default=None, description="参数键值对(JSON或对象)")
    userId: int | None = Field(default=None, description="所属用户ID")
    isDefault: int = Field(description="是否默认预设")
    createTime: datetime | None = Field(default=None, description="创建时间")
