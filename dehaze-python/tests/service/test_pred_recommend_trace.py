"""预测/评估日志追踪链路测试（真实 MySQL 测试库 + SAVEPOINT 回滚）。

覆盖：
- A2 配额回滚：predict 扣配额后前置失败（网络错/SSRF 拒绝/文件不存在）归还配额
- C2 推荐采纳追踪：recommendedBy 落库 sys_pred_log.recommended_by，
  推荐报表采纳率口径 = 带推荐来源的预测记录数 / 推荐总数
- C3 任务类型：sys_eval_log.task_type 写入（evaluation/report），
  GET /evaluation/metrics 仅返回 evaluation 行，不混入报告行

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_log import SysEvalLog
from app.models.entity.sys_recommendation import SysRecommendation
from app.models.enum.log_status import LogStatus
from app.repository.pred_eval_log_repository import (
    eval_log_repository,
    pred_log_repository,
)
from app.repository.recommendation_repository import recommendation_repository
from app.service.compare_service import compare_service
from app.service.evaluation_service import evaluation_service
from app.service.member.quota_service import member_quota_service
from app.service.prediction import prediction_service as prediction_module
from app.service.prediction.prediction_service import prediction_service
from app.service.recommendation_service import recommendation_service

pytestmark = pytest.mark.requires_db

USER_A = 2101001

_MD5 = "a" * 32


# ===== 公共夹具 =====


@pytest.fixture
def quota_mocks(monkeypatch):
    deduct = AsyncMock()
    restore = AsyncMock()
    monkeypatch.setattr(member_quota_service, "check_and_deduct_quota", deduct)
    monkeypatch.setattr(member_quota_service, "restore_quota", restore)
    return deduct, restore


def _patch_algorithm_ok(monkeypatch):
    async def ok_algorithm(algorithm_id):
        return SimpleNamespace(id=algorithm_id, name="测试算法")

    monkeypatch.setattr(prediction_service, "get_algorithm", ok_algorithm)


# ===== A2 配额回滚 =====


async def test_predict_quota_restored_when_image_download_fails(db, quota_mocks, monkeypatch):
    """扣配额后图片下载网络错误 → 配额归还，异常原样抛出"""
    deduct, restore = quota_mocks
    _patch_algorithm_ok(monkeypatch)
    monkeypatch.setattr(prediction_service, "_execute_async", AsyncMock())

    async def fail_fetch(url):
        raise RuntimeError("网络超时")

    monkeypatch.setattr(prediction_module, "fetch_image", fail_fetch)

    with pytest.raises(RuntimeError):
        await prediction_service.predict(
            algorithm_id=1,
            image_url="https://cdn.example.com/a.png",
            user_id=USER_A,
        )

    deduct.assert_awaited_once()
    restore.assert_awaited_once()


async def test_predict_quota_restored_when_ssrf_rejected(db, quota_mocks, monkeypatch):
    """扣配额后图片地址触发 SSRF 防护拒绝 → 配额归还"""
    deduct, restore = quota_mocks
    _patch_algorithm_ok(monkeypatch)
    monkeypatch.setattr(prediction_service, "_execute_async", AsyncMock())

    async def ssrf_reject(url):
        raise BusinessException(ResultCode.PARAM_ERROR, "图片地址不允许访问内网资源")

    monkeypatch.setattr(prediction_module, "fetch_image", ssrf_reject)

    with pytest.raises(BusinessException):
        await prediction_service.predict(
            algorithm_id=1,
            image_url="http://169.254.169.254/latest/meta-data",
            user_id=USER_A,
        )

    deduct.assert_awaited_once()
    restore.assert_awaited_once()


async def test_predict_quota_restored_when_file_not_found(db, quota_mocks, monkeypatch):
    """扣配额后 fileId 不存在 → 配额归还"""
    deduct, restore = quota_mocks
    _patch_algorithm_ok(monkeypatch)

    from app.repository import file_repository

    monkeypatch.setattr(file_repository.file_repository, "get_by_id", AsyncMock(return_value=None))

    with pytest.raises(BusinessException) as ei:
        await prediction_service.predict(
            algorithm_id=1,
            image_url="",
            user_id=USER_A,
            file_id=99999999,
        )
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND

    deduct.assert_awaited_once()
    restore.assert_awaited_once()


async def test_predict_m2m_failure_neither_deducts_nor_restores(db, quota_mocks, monkeypatch):
    """m2m 免配额用户（skip_quota_check）前置失败：不扣减也不归还"""
    deduct, restore = quota_mocks
    _patch_algorithm_ok(monkeypatch)

    async def fail_fetch(url):
        raise RuntimeError("网络超时")

    monkeypatch.setattr(prediction_module, "fetch_image", fail_fetch)

    with pytest.raises(RuntimeError):
        await prediction_service.predict(
            algorithm_id=1,
            image_url="https://cdn.example.com/a.png",
            user_id=USER_A,
            skip_quota_check=True,
        )

    deduct.assert_not_awaited()
    restore.assert_not_awaited()


async def test_predict_success_keeps_quota_deducted(db, quota_mocks, monkeypatch):
    """回归：预测正常提交（processing）不归还配额"""
    deduct, restore = quota_mocks
    _patch_algorithm_ok(monkeypatch)
    monkeypatch.setattr(prediction_service, "_execute_async", AsyncMock())

    async def ok_fetch(url):
        return BytesIO(b"fake-png")

    monkeypatch.setattr(prediction_module, "fetch_image", ok_fetch)

    result = await prediction_service.predict(
        algorithm_id=1,
        image_url="https://cdn.example.com/a.png",
        user_id=USER_A,
    )

    assert result["status"] == LogStatus.PROCESSING.value
    deduct.assert_awaited_once()
    restore.assert_not_awaited()


# ===== C2 recommended_by 落库 =====


async def test_predict_persists_recommended_by(db, quota_mocks, monkeypatch):
    """predict 接收 recommended_by 并落库到 sys_pred_log.recommended_by"""
    _patch_algorithm_ok(monkeypatch)
    monkeypatch.setattr(prediction_service, "_execute_async", AsyncMock())

    async def ok_fetch(url):
        return BytesIO(b"fake-png")

    monkeypatch.setattr(prediction_module, "fetch_image", ok_fetch)

    result = await prediction_service.predict(
        algorithm_id=1,
        image_url="https://cdn.example.com/a.png",
        user_id=USER_A,
        recommended_by=888001,
    )

    log = await pred_log_repository.get_by_id(db, result["logId"])
    assert log is not None
    assert log.recommended_by == 888001


async def test_cache_hit_completed_log_persists_recommended_by(db):
    """缓存命中路径（create_log）同样落库 recommended_by"""
    log = await pred_log_repository.create_log(
        db=db,
        algorithm_id=1,
        origin_md5=_MD5,
        origin_url="https://cdn.example.com/a.png",
        pred_md5=_MD5,
        pred_url="https://storage/result.png",
        time_ms=100,
        recommended_by=888002,
    )
    reloaded = await pred_log_repository.get_by_id(db, log.id)
    assert reloaded is not None
    assert reloaded.recommended_by == 888002


async def test_predict_without_recommendation_keeps_null(db, quota_mocks, monkeypatch):
    """不传 recommended_by 时落库为 NULL（不写占位值）"""
    _patch_algorithm_ok(monkeypatch)
    monkeypatch.setattr(prediction_service, "_execute_async", AsyncMock())

    async def ok_fetch(url):
        return BytesIO(b"fake-png")

    monkeypatch.setattr(prediction_module, "fetch_image", ok_fetch)

    result = await prediction_service.predict(
        algorithm_id=1,
        image_url="https://cdn.example.com/a.png",
        user_id=USER_A,
    )

    log = await pred_log_repository.get_by_id(db, result["logId"])
    assert log is not None
    assert log.recommended_by is None


# ===== C2 推荐报表采纳口径 =====


async def _create_recommendation(db, md5: str, feedback: int = 0) -> int:
    rec = SysRecommendation(
        user_id=USER_A,
        image_md5=md5,
        target_type="algorithm",
        top_algorithms=[{"algorithmId": 1, "algorithmName": "测试算法", "matchScore": 60}],
        feedback=feedback,
    )
    created = await recommendation_repository.create(db, rec)
    return created.id


async def _create_pred_log(db, recommended_by: int | None = None) -> int:
    from app.models.base import set_current_user_id

    set_current_user_id(USER_A)
    try:
        log = await pred_log_repository.create_log(
            db=db,
            algorithm_id=1,
            origin_md5=_MD5,
            origin_url="https://cdn.example.com/a.png",
            pred_md5=_MD5,
            pred_url="https://storage/result.png",
            time_ms=100,
            recommended_by=recommended_by,
        )
    finally:
        set_current_user_id(None)
    return log.id


async def test_report_adoption_rate_counts_recommended_predictions(db):
    """采纳率 = 带推荐来源的预测记录数 / 推荐总数；满意度仍为有用反馈占比"""
    await _create_recommendation(db, "md5-report-1", feedback=1)
    await _create_recommendation(db, "md5-report-2", feedback=2)
    await _create_recommendation(db, "md5-report-3", feedback=0)
    await _create_pred_log(db, recommended_by=1)
    await _create_pred_log(db, recommended_by=1)
    await _create_pred_log(db, recommended_by=None)

    report = await recommendation_service.get_report(db, None, None)

    assert report.totalRecommendations == 3
    assert report.adoptionRate == pytest.approx(2 / 3)
    assert report.satisfactionRate == pytest.approx(1 / 2)


async def test_report_trend_uses_recommended_prediction_ratio(db):
    """趋势按日聚合：当日带推荐来源预测数 / 当日推荐总数"""
    await _create_recommendation(db, "md5-trend-1")
    await _create_recommendation(db, "md5-trend-2")
    await _create_pred_log(db, recommended_by=1)

    report = await recommendation_service.get_report(db, None, None)

    assert len(report.trend) == 1
    assert report.trend[0].adoptionRate == pytest.approx(1 / 2)


async def test_report_no_recommendation_zero_adoption(db):
    """无推荐记录时采纳率为 0，不抛除零错误"""
    await _create_pred_log(db, recommended_by=1)

    report = await recommendation_service.get_report(db, None, None)

    assert report.totalRecommendations == 0
    assert report.adoptionRate == 0.0


# ===== C3 task_type 写入与过滤 =====


async def test_eval_pending_log_writes_task_type_evaluation(db):
    """评估任务创建日志 task_type=evaluation"""
    log = await eval_log_repository.create_pending_log(
        db=db,
        algorithm_id=1,
        pred_md5=_MD5,
        pred_url="https://storage/pred.png",
        gt_md5="b" * 32,
        gt_url="https://storage/gt.png",
    )
    reloaded = await eval_log_repository.get_by_id(db, log.id)
    assert reloaded is not None
    assert reloaded.task_type == "evaluation"


async def test_compare_report_log_writes_task_type_report(db, monkeypatch):
    """对比报告任务创建日志 task_type=report"""
    pred_log = await _create_pred_log(db, recommended_by=None)

    async def ok_algorithm(algorithm_id):
        return SimpleNamespace(id=1, name="测试算法")

    monkeypatch.setattr(prediction_service, "get_algorithm", ok_algorithm)
    monkeypatch.setattr(compare_service, "_generate_async", AsyncMock())

    result = await compare_service.generate_report(pred_log, USER_A)

    report_log = await eval_log_repository.get_by_id(db, result["taskId"])
    assert report_log is not None
    assert report_log.task_type == "report"


async def test_metrics_excludes_report_rows(db):
    """/evaluation/metrics 仅返回 task_type=evaluation 行，报告行不混入"""
    eval_log = SysEvalLog(
        algorithm_id=1,
        pred_md5=_MD5,
        pred_url="https://storage/pred.png",
        gt_md5="b" * 32,
        gt_url="https://storage/gt.png",
        time=1,
        status=LogStatus.COMPLETED.value,
        task_type="evaluation",
        result={"PSNR": 35.0},
        create_by=USER_A,
    )
    report_log = SysEvalLog(
        algorithm_id=1,
        pred_md5=_MD5,
        pred_url="https://storage/pred.png",
        gt_md5="c" * 32,
        gt_url="https://storage/gt.png",
        time=1,
        status=LogStatus.COMPLETED.value,
        task_type="report",
        result={"reportHtml": "<html/>"},
        create_by=USER_A,
    )
    await eval_log_repository.create(db, eval_log)
    await eval_log_repository.create(db, report_log)

    logs, total = await evaluation_service.list_completed_metrics(
        db, user_id=USER_A, page=1, size=10
    )

    assert total == 1
    assert [log.task_type for log in logs] == ["evaluation"]
