"""预测模块 Schema 模型"""

from pydantic import BaseModel, Field

# ==================== 表单参数模型 ====================


class BatchPredictionItem(BaseModel):
    """批量预测单项（fileId 与 imageUrl 至少提供一个）"""

    fileId: int | None = Field(default=None, description="原始图片文件ID")
    imageUrl: str | None = Field(default=None, description="原始图片URL（与fileId二选一）")
    params: str | None = Field(default=None, description="预测参数(JSON)")
