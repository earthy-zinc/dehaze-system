"""
对比报告 API 路由 —— 生成/下载去雾效果对比报告（异步任务模式）

POST /api/v1/compare/report          → 提交对比报告生成任务，返回 {taskId, status: "processing"}
GET  /api/v1/compare/report/{taskId} → 查询状态（默认）或下载 HTML 文件流（?download=true）
"""

import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.compare import CompareReportForm, CompareReportResultVO
from app.service.compare_service import compare_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/compare", tags=["效果对比"], dependencies=[Depends(get_current_user)]
)


@router.post(
    "/report", response_model=Result[CompareReportResultVO], summary="生成对比报告（异步任务）"
)
async def generate_report(
    body: CompareReportForm,
    user: UserContext = Depends(get_current_user),
):
    """
    提交对比报告生成任务（异步）

    立即返回 taskId + status=processing，通过 GET /report/{taskId} 轮询结果
    """
    result = await compare_service.generate_report(
        log_id=body.logId,
        user_id=user.id,
    )
    return success(
        CompareReportResultVO(
            taskId=result["taskId"],
            status=result["status"],
        )
    )


@router.get("/report/{task_id}", summary="查询对比报告状态/下载对比报告")
async def get_or_download_report(
    task_id: int,
    download: bool = Query(
        default=False, description="下载标识（true时返回HTML文件流，否则返回JSON状态）"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    查询对比报告任务状态，或下载已完成的报告 HTML 文件

    - download=false（默认）：返回 JSON 格式任务状态
    - download=true：返回 HTML 文件流（仅 completed 状态可用）
    """
    if not download:
        status_data = await compare_service.get_report_status(task_id)
        return success(
            CompareReportResultVO(
                taskId=status_data["taskId"],
                status=status_data["status"],
                downloadUrl=status_data.get("downloadUrl"),
                errorMessage=status_data.get("errorMessage"),
            )
        )

    report_html = await compare_service.get_report_html(task_id)
    return HTMLResponse(
        content=report_html,
        status_code=200,
        headers={"Content-Disposition": "inline; filename=compare-report.html"},
    )
