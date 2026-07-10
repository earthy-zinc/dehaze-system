"""
导出任务模块 Schema 模型
"""
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field

from app.models.enum.task_enum import TaskType


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
    type: TaskType = Field(
        ...,
        description="导出类型：dataset_export, item_download, batch_download, custom_export"
    )
    targetId: Optional[int] = Field(
        default=None,
        description="单个导出目标ID（type为dataset_export或item_download时使用）"
    )
    targetIds: Optional[List[int]] = Field(
        default=None,
        description="批量导出目标ID列表（type为batch_download或custom_export时使用）"
    )
    options: Optional[ExportOptions] = Field(
        default=None,
        description="导出选项"
    )


class TaskListQuery(BaseModel):
    """任务列表查询参数"""
    status: Optional[str] = Field(default=None, description="状态筛选")
    taskType: Optional[str] = Field(default=None, description="类型筛选")
    page: int = Field(default=1, ge=1, description="页码")
    size: int = Field(default=10, ge=1, le=100, description="每页数量")


# ==================== 响应模型 ====================

class TaskVO(BaseModel):
    """任务VO"""
    id: int = Field(description="任务主键ID")
    taskId: str = Field(description="任务ID（UUID）")
    taskType: str = Field(description="任务类型")
    status: str = Field(description="任务状态：pending, processing, completed, failed, cancelled")
    progress: int = Field(description="执行进度(0-100)")
    totalFiles: int = Field(default=0, description="总文件数")
    processedFiles: Optional[int] = Field(default=0, description="已处理文件数")
    result: Optional[str] = Field(default=None, description="任务结果")
    downloadUrl: Optional[str] = Field(default=None, description="下载链接")
    error: Optional[str] = Field(default=None, description="错误信息")
    createdAt: Optional[datetime] = Field(default=None, description="创建时间")
    updatedAt: Optional[datetime] = Field(default=None, description="更新时间")
    startedAt: Optional[datetime] = Field(default=None, description="开始时间")
    completedAt: Optional[datetime] = Field(default=None, description="完成时间")
    expiresAt: Optional[datetime] = Field(default=None, description="过期时间")


class TaskPageVO(BaseModel):
    """任务分页结果"""
    list: List[TaskVO] = Field(description="任务列表")
    total: int = Field(description="总数")
