"""
任务模块 Schema 模型
"""
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field

from app.models.enum.task_enum import TaskType


class TaskIdPath(BaseModel):
    task_id: str = Field(..., description="任务ID（UUID格式）")


class ExportTaskCreateForm(BaseModel):
    type: TaskType = Field(..., description="任务类型")
    params_json: Optional[str] = Field(default=None, description="任务参数（JSON 字符串）")


class TaskVO(BaseModel):
    id: int = Field(description="任务主键ID")
    taskId: str = Field(description="任务ID（UUID）")
    taskType: str = Field(description="任务类型")
    status: int = Field(description="任务状态(1:待处理;2:处理中;3:已完成;4:失败;5:已取消)")
    progress: int = Field(description="执行进度(0-100)")
    totalFiles: int = Field(default=0, description="总文件数")
    processedFiles: Optional[int] = Field(default=0, description="已处理文件数")
    downloadUrl: Optional[str] = Field(default=None, description="下载链接")
    error: Optional[str] = Field(default=None, description="错误信息")
    createdAt: Optional[datetime] = Field(default=None, description="创建时间")
    startedAt: Optional[datetime] = Field(default=None, description="开始时间")
    completedAt: Optional[datetime] = Field(default=None, description="完成时间")
    expiresAt: Optional[datetime] = Field(default=None, description="过期时间")
    idempotencyKey: Optional[str] = Field(default=None, description="客户端幂等键")
    retryCount: int = Field(default=0, description="MQ 重试次数")
    workerId: Optional[str] = Field(default=None, description="执行 Worker 标识")


class TaskPageVO(BaseModel):
    list: List[TaskVO] = Field(description="任务列表")
    total: int = Field(description="总数")


class ExportTaskVO(BaseModel):
    taskId: str = Field(description="任务ID")
    status: int = Field(description="任务状态(1:待处理;2:处理中;3:已完成;4:失败;5:已取消)")
    estimatedCount: int = Field(default=0, description="预估数据量")


class ImportTaskVO(BaseModel):
    taskId: str = Field(description="任务ID")
    status: int = Field(description="任务状态(1:待处理;2:处理中;3:已完成;4:失败;5:已取消)")


class ImportErrorVO(BaseModel):
    row: int = Field(description="行号")
    field: Optional[str] = Field(default=None, description="字段名")
    message: str = Field(description="错误信息")


class ImportResultVO(BaseModel):
    totalRows: int = Field(description="总行数")
    successCount: int = Field(description="成功数")
    failureCount: int = Field(description="失败数")
    skippedCount: int = Field(default=0, description="跳过数")
    errors: List[ImportErrorVO] = Field(default_factory=list, description="错误明细")
    errorReportUrl: Optional[str] = Field(default=None, description="错误报告下载链接")
