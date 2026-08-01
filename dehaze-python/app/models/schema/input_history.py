"""
图像输入历史记录 Schema
对齐 dehaze-java HistoryForm/HistoryUpdateForm/InputHistoryVO 字段命名
"""
from typing import Optional

from pydantic import BaseModel, Field


class InputHistoryForm(BaseModel):
    """历史记录创建表单 (对齐 Java HistoryForm 字段)"""
    originalImageUrl: Optional[str] = Field(default=None, alias="originalImageUrl", description="原始图片URL")
    originalThumbnailUrl: Optional[str] = Field(default=None, alias="originalThumbnailUrl", description="原始缩略图URL")
    resultImageUrl: Optional[str] = Field(default=None, alias="resultImageUrl", description="处理结果图片URL")
    resultThumbnailUrl: Optional[str] = Field(default=None, alias="resultThumbnailUrl", description="结果缩略图URL")
    algorithmId: Optional[int] = Field(default=None, alias="algorithmId", description="算法ID")
    algorithmName: Optional[str] = Field(default=None, alias="algorithmName", description="算法名称（冗余）")
    algorithmParams: Optional[str] = Field(default=None, alias="algorithmParams", description="算法参数（JSON）")
    processingTime: Optional[int] = Field(default=None, alias="processingTime", description="处理耗时（毫秒）")
    status: Optional[int] = Field(default=3, description="处理状态（1=成功，2=失败，3=处理中）")
    inputSource: Optional[str] = Field(default=None, alias="inputSource", description="图片来源（upload/camera/sample）")

    model_config = {"populate_by_name": True}


class InputHistoryVO(BaseModel):
    """历史记录VO (对齐 Java InputHistoryVO 字段)"""
    id: int = Field(description="记录ID")
    userId: Optional[int] = Field(default=None, validation_alias="user_id", serialization_alias="userId", description="用户ID")
    originalImageUrl: Optional[str] = Field(default=None, validation_alias="original_image_url", serialization_alias="originalImageUrl", description="原始图片URL")
    originalThumbnailUrl: Optional[str] = Field(default=None, validation_alias="original_thumbnail_url", serialization_alias="originalThumbnailUrl", description="原始缩略图URL")
    resultImageUrl: Optional[str] = Field(default=None, validation_alias="result_image_url", serialization_alias="resultImageUrl", description="处理结果图片URL")
    resultThumbnailUrl: Optional[str] = Field(default=None, validation_alias="result_thumbnail_url", serialization_alias="resultThumbnailUrl", description="结果缩略图URL")
    algorithmId: Optional[int] = Field(default=None, validation_alias="algorithm_id", serialization_alias="algorithmId", description="算法ID")
    algorithmName: Optional[str] = Field(default=None, validation_alias="algorithm_name", serialization_alias="algorithmName", description="算法名称")
    algorithmParams: Optional[str] = Field(default=None, validation_alias="algorithm_params", serialization_alias="algorithmParams", description="算法参数（JSON）")
    processingTime: Optional[int] = Field(default=None, validation_alias="processing_time", serialization_alias="processingTime", description="处理耗时（毫秒）")
    status: Optional[int] = Field(default=None, description="处理状态（1=成功，2=失败，3=处理中）")
    inputSource: Optional[str] = Field(default=None, validation_alias="input_source", serialization_alias="inputSource", description="图片来源")
    syncStatus: Optional[int] = Field(default=0, validation_alias="sync_status", serialization_alias="syncStatus", description="同步状态")
    createTime: Optional[str] = Field(default=None, validation_alias="create_time", serialization_alias="createTime", description="创建时间")
    updateTime: Optional[str] = Field(default=None, validation_alias="update_time", serialization_alias="updateTime", description="更新时间")

    model_config = {"populate_by_name": True}
