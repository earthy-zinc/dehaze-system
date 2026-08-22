"""
图像输入历史记录 Schema
对齐 dehaze-java HistoryForm/InputHistoryVO 字段命名
"""

from pydantic import BaseModel, Field


class InputHistoryForm(BaseModel):
    """历史记录创建表单 (对齐 Java HistoryForm 字段)"""

    originalImageUrl: str | None = Field(
        default=None, alias="originalImageUrl", description="原始图片URL"
    )
    originalThumbnailUrl: str | None = Field(
        default=None, alias="originalThumbnailUrl", description="原始缩略图URL"
    )
    resultImageUrl: str | None = Field(
        default=None, alias="resultImageUrl", description="处理结果图片URL"
    )
    resultThumbnailUrl: str | None = Field(
        default=None, alias="resultThumbnailUrl", description="结果缩略图URL"
    )
    algorithmId: int | None = Field(default=None, alias="algorithmId", description="算法ID")
    algorithmName: str | None = Field(
        default=None, alias="algorithmName", description="算法名称（冗余）"
    )
    algorithmParams: str | None = Field(
        default=None, alias="algorithmParams", description="算法参数（JSON）"
    )
    processingTime: int | None = Field(
        default=None, alias="processingTime", description="处理耗时（毫秒）"
    )
    status: int | None = Field(
        default=3, description="处理状态（1=成功，2=失败，3=处理中），创建时确定"
    )
    inputSource: str | None = Field(
        default=None, alias="inputSource", description="图片来源（upload/camera/sample）"
    )

    model_config = {"populate_by_name": True}


class InputHistoryVO(BaseModel):
    """历史记录VO (对齐 Java InputHistoryVO 字段)"""

    id: int = Field(description="记录ID")
    userId: int | None = Field(
        default=None, validation_alias="user_id", serialization_alias="userId", description="用户ID"
    )
    originalImageUrl: str | None = Field(
        default=None,
        validation_alias="original_image_url",
        serialization_alias="originalImageUrl",
        description="原始图片URL",
    )
    originalThumbnailUrl: str | None = Field(
        default=None,
        validation_alias="original_thumbnail_url",
        serialization_alias="originalThumbnailUrl",
        description="原始缩略图URL",
    )
    resultImageUrl: str | None = Field(
        default=None,
        validation_alias="result_image_url",
        serialization_alias="resultImageUrl",
        description="处理结果图片URL",
    )
    resultThumbnailUrl: str | None = Field(
        default=None,
        validation_alias="result_thumbnail_url",
        serialization_alias="resultThumbnailUrl",
        description="结果缩略图URL",
    )
    algorithmId: int | None = Field(
        default=None,
        validation_alias="algorithm_id",
        serialization_alias="algorithmId",
        description="算法ID",
    )
    algorithmName: str | None = Field(
        default=None,
        validation_alias="algorithm_name",
        serialization_alias="algorithmName",
        description="算法名称",
    )
    algorithmParams: str | None = Field(
        default=None,
        validation_alias="algorithm_params",
        serialization_alias="algorithmParams",
        description="算法参数（JSON）",
    )
    processingTime: int | None = Field(
        default=None,
        validation_alias="processing_time",
        serialization_alias="processingTime",
        description="处理耗时（毫秒）",
    )
    status: int | None = Field(
        default=None,
        description="处理状态（1=成功，2=失败，3=处理中），创建时确定，不随处理进度更新",
    )
    inputSource: str | None = Field(
        default=None,
        validation_alias="input_source",
        serialization_alias="inputSource",
        description="图片来源",
    )
    createTime: str | None = Field(
        default=None,
        validation_alias="create_time",
        serialization_alias="createTime",
        description="创建时间",
    )
    updateTime: str | None = Field(
        default=None,
        validation_alias="update_time",
        serialization_alias="updateTime",
        description="更新时间",
    )

    model_config = {"populate_by_name": True}
