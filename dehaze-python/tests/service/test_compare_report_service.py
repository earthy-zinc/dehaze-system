"""对比报告服务测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖：报告任务归属校验（处理记录/报告日志越权按不存在处理）、未完成记录拦截、
状态机口径（processing/failed/内容为空）、报告 HTML 动态字段 XSS 转义。

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_log import SysEvalLog, SysPredLog
from app.models.enum.log_status import LogStatus
from app.repository.pred_eval_log_repository import eval_log_repository, pred_log_repository
from app.service.compare_service import compare_service

pytestmark = pytest.mark.requires_db

USER_A = 2003001
USER_B = 2003002


async def _create_pred_log(db, user_id: int, status: int = LogStatus.COMPLETED.value) -> SysPredLog:
    log = SysPredLog(
        algorithm_id=1,
        origin_md5="c" * 32,
        origin_url="http://storage/origin.jpg",
        pred_md5="d" * 32,
        pred_url="http://storage/pred.jpg",
        time=2,
        status=status,
        create_by=user_id,
    )
    return await pred_log_repository.create(db, log)


async def _create_report_log(
    db, user_id: int, status: int, result=None, error_message: str | None = None
) -> SysEvalLog:
    log = SysEvalLog(
        algorithm_id=1,
        pred_md5="",
        pred_url="http://storage/pred.jpg",
        gt_md5="",
        gt_url="http://storage/origin.jpg",
        time=0,
        status=status,
        result=result,
        error_message=error_message,
        create_by=user_id,
    )
    return await eval_log_repository.create(db, log)


@pytest.fixture(autouse=True)
def _no_background_generation(monkeypatch):
    """拦截后台 HTML 生成任务与算法查询，避免测试依赖种子数据/真实写库"""
    monkeypatch.setattr(type(compare_service), "_generate_async", AsyncMock(), raising=False)

    import app.service.prediction.prediction_service as prediction_module

    algorithm_stub = AsyncMock(return_value=SimpleNamespace(name="测试算法"))
    monkeypatch.setattr(prediction_module.prediction_service, "get_algorithm", algorithm_stub)


# ===== 报告任务归属校验 =====


async def test_generate_report_rejects_other_users_pred_log(db):
    """用他人处理记录生成报告 → A0401（按不存在处理，不泄露资源存在性）"""
    other_pred = await _create_pred_log(db, USER_A)

    with pytest.raises(BusinessException) as exc_info:
        await compare_service.generate_report(log_id=other_pred.id, user_id=USER_B)
    assert exc_info.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_generate_report_rejects_incomplete_pred_log(db):
    """处理记录未完成 → A0500"""
    pred = await _create_pred_log(db, USER_A, status=LogStatus.PROCESSING.value)

    with pytest.raises(BusinessException) as exc_info:
        await compare_service.generate_report(log_id=pred.id, user_id=USER_A)
    assert exc_info.value.code == ResultCode.BUSINESS_ERROR


async def test_generate_report_creates_processing_task_for_owner(db):
    """本人已完成记录 → 创建 processing 报告任务并返回 taskId"""
    pred = await _create_pred_log(db, USER_A)

    result = await compare_service.generate_report(log_id=pred.id, user_id=USER_A)

    assert result["taskId"] > 0
    assert result["status"] == LogStatus.PROCESSING.value


async def test_get_report_status_rejects_other_users_task(db):
    """查询他人报告任务 → A0401"""
    report = await _create_report_log(db, USER_A, status=LogStatus.COMPLETED.value, result={})

    with pytest.raises(BusinessException) as exc_info:
        await compare_service.get_report_status(report.id, user_id=USER_B)
    assert exc_info.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_get_report_status_completed_returns_download_url(db):
    """本人已完成报告 → 返回 downloadUrl"""
    report = await _create_report_log(db, USER_A, status=LogStatus.COMPLETED.value, result={})

    data = await compare_service.get_report_status(report.id, user_id=USER_A)

    assert data["status"] == LogStatus.COMPLETED.value
    assert data["downloadUrl"] == f"/api/v1/compare/report/{report.id}?download=true"


async def test_get_report_html_rejects_other_users_task(db):
    """下载他人报告 → A0401（即使是已完成的报告）"""
    report = await _create_report_log(
        db, USER_A, status=LogStatus.COMPLETED.value, result={"reportHtml": "<html/>"}
    )

    with pytest.raises(BusinessException) as exc_info:
        await compare_service.get_report_html(report.id, user_id=USER_B)
    assert exc_info.value.code == ResultCode.RESOURCE_NOT_FOUND


# ===== 报告状态机口径 =====


async def test_get_report_html_returns_content_for_owner(db):
    """本人已完成报告 → 返回 HTML 内容"""
    report = await _create_report_log(
        db, USER_A, status=LogStatus.COMPLETED.value, result={"reportHtml": "<html>ok</html>"}
    )

    assert await compare_service.get_report_html(report.id, user_id=USER_A) == "<html>ok</html>"


async def test_get_report_html_processing_rejected(db):
    """processing 报告不可下载 → A0500"""
    report = await _create_report_log(db, USER_A, status=LogStatus.PROCESSING.value)

    with pytest.raises(BusinessException) as exc_info:
        await compare_service.get_report_html(report.id, user_id=USER_A)
    assert exc_info.value.code == ResultCode.BUSINESS_ERROR


async def test_get_report_html_failed_reports_error(db):
    """failed 报告下载 → B0001，带失败原因"""
    report = await _create_report_log(
        db, USER_A, status=LogStatus.FAILED.value, error_message="渲染失败"
    )

    with pytest.raises(BusinessException) as exc_info:
        await compare_service.get_report_html(report.id, user_id=USER_A)
    assert exc_info.value.code == ResultCode.SYSTEM_EXECUTION_ERROR


async def test_get_report_html_empty_content_rejected(db):
    """已完成但报告内容为空 → A0401"""
    report = await _create_report_log(db, USER_A, status=LogStatus.COMPLETED.value, result=None)

    with pytest.raises(BusinessException) as exc_info:
        await compare_service.get_report_html(report.id, user_id=USER_A)
    assert exc_info.value.code == ResultCode.RESOURCE_NOT_FOUND


# ===== 报告 HTML XSS 转义 =====


def test_build_report_html_escapes_dynamic_fields():
    """算法名与图片 URL 中的 HTML 特殊字符被转义，不构成存储型 XSS"""
    html = compare_service._build_report_html(
        algorithm_name="<script>alert(1)</script>",
        generated_at="2026-09-12 00:00:00",
        origin_url='http://x/a.jpg" onerror="alert(1)',
        result_url="http://x/b.jpg",
        algorithm_id=1,
    )

    assert "<script>alert(1)</script>" not in html
    assert 'onerror="alert(1)' not in html
    assert "&lt;script&gt;" in html
    assert "&quot; onerror=&quot;alert(1)" in html
