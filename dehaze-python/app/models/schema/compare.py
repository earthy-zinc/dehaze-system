"""
对比报告 Schema 模型
"""

from pydantic import BaseModel, Field


class CompareReportForm(BaseModel):
    """对比报告生成请求"""

    logId: int = Field(description="处理日志ID（sys_pred_log.id）")
    format: str = Field(description="报告格式：pdf 或 image")
    includeMetrics: bool | None = Field(default=None, description="是否包含评估指标")
    includeFilters: bool | None = Field(default=None, description="是否包含滤镜参数")


class CompareReportResultVO(BaseModel):
    """对比报告任务状态响应（POST 返回 + GET 查询状态）"""

    taskId: int | None = Field(default=None, description="报告任务ID（即 sys_eval_log.id）")
    status: int = Field(description="任务状态(1:处理中;2:已完成;3:失败)")
    downloadUrl: str | None = Field(default=None, description="下载链接（completed 时返回）")
    errorMessage: str | None = Field(default=None, description="失败错误信息（failed 时返回）")
