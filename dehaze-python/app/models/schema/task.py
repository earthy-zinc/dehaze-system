"""
导出任务模块 Schema 模型
"""
from typing import Optional, List
from pydantic import BaseModel, Field


# ==================== 路径参数模型 ====================

class TaskIdPath(BaseModel):
    """任务ID路径参数"""
    task_id: str = Field(..., description="任务ID（UUID格式）")


# ==================== 请求体模型 ====================

class ExportOptions(BaseModel):
    """导出选项"""
    structure: Optional[str] = Field(
        default='by_item',
        description="文件组织结构：by_item-按数据项组织, by_type-按文件类型组织"
    )
    includeTypes: Optional[List[str]] = Field(
        default=None,
        description="包含的文件类型（不传则包含所有）"
    )
    includeThumbnail: Optional[bool] = Field(
        default=False,
        description="是否包含缩略图"
    )


class ExportTaskCreateForm(BaseModel):
    """导出任务创建表单"""
    type: str = Field(
        ...,
        description="导出类型：dataset-单个数据集, dataset_item-单个数据项, batch_items-批量数据项, custom-自定义"
    )
    targetId: Optional[int] = Field(
        default=None,
        description="单个导出目标ID（type为dataset或dataset_item时使用）"
    )
    targetIds: Optional[List[int]] = Field(
        default=None,
        description="批量导出目标ID列表（type为batch_items或custom时使用）"
    )
    options: Optional[ExportOptions] = Field(
        default=None,
        description="导出选项"
    )


# ==================== 响应模型 ====================

class TaskVO(BaseModel):
    """任务VO"""
    id: int = Field(description="任务主键ID")
    taskId: str = Field(description="任务ID（UUID）")
    taskType: str = Field(description="任务类型")
    status: str = Field(description="任务状态：pending, processing, completed, failed, cancelled")
    progress: int = Field(description="执行进度(0-100)")
    totalFiles: int = Field(description="总文件数")
    processedFiles: Optional[int] = Field(default=None, description="已处理文件数")
    downloadUrl: Optional[str] = Field(default=None, description="下载链接")
    error: Optional[str] = Field(default=None, description="错误信息")
    createdAt: Optional[str] = Field(default=None, description="创建时间")
    startedAt: Optional[str] = Field(default=None, description="开始时间")
    completedAt: Optional[str] = Field(default=None, description="完成时间")
    expiresAt: Optional[str] = Field(default=None, description="过期时间")
