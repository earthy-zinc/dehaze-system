"""预测/评估日志追踪链路测试（真实 MySQL 测试库 + SAVEPOINT 回滚，统一模型口径）。

覆盖：
- C2 推荐采纳追踪：recommendedBy 落库 sys_pred_log.recommended_by，
  推荐报表采纳率口径 = 带推荐来源的预测记录数 / 推荐总数
- C3 任务类型：sys_eval_log.task_type 写入（evaluation/report），
  GET /evaluation/metrics 仅返回 evaluation 行，不混入报告行

遵循 clarity 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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
from app.service.prediction import prediction_service as prediction_module
from app.service.prediction.prediction_service import prediction_service
from app.service.recommendation_service import recommendation_service

pytestmark = pytest.mark.requires_db

USER_A = 2101001
MODEL_ID = "aecrnet"

_MD5 = "a" * 32


# ===== 公共夹具 =====


@pytest.fixture
def no_balance_check(monkeypatch):
    """跳过调用前余额预校验（本组用例只关心推荐来源/计费无关的落库口径）"""
    monkeypatch.setattr(prediction_module.image_billing_service, "ensure_balance", AsyncMock())


def _patch_model_ok(monkeypatch):
    async def ok_model(model_id):
        return SimpleNamespace(id=1, model_id=model_id, display_name="AECR-NET")

    monkeypatch.setattr(prediction_service, "get_model", ok_model)


def _patch_fetch_ok(monkeypatch):
    async def ok_fetch(url):
        return BytesIO(b"fake-png")

    monkeypatch.setattr(prediction_module, "fetch_image", ok_fetch)


# ===== C2 recommended_by 落库 =====


async def test_predict_persists_recommended_by(db, no_balance_check, monkeypatch):
    """predict 接收 recommended_by 并落库到 sys_pred_log.recommended_by"""
    _patch_model_ok(monkeypatch)
    _patch_fetch_ok(monkeypatch)
    monkeypatch.setattr(prediction_service, "_execute_async", AsyncMock())

    result = await prediction_service.predict(
        model=MODEL_ID,
        image_url="https://cdn.example.com/a.png",
        user_id=USER_A,
        recommended_by=888001,
    )

    log = await pred_log_repository.get_by_id(db, result["taskId"])
    assert log is not None
    assert log.recommended_by == 888001


async def test_cache_hit_completed_log_persists_recommended_by(db):
    """缓存命中路径（create_log）同样落库 recommended_by"""
    log = await pred_log_repository.create_log(
        db=db,
        model_id=MODEL_ID,
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


async def test_predict_without_recommendation_keeps_null(db, no_balance_check, monkeypatch):
    """不传 recommended_by 时落库为 NULL（不写占位值）"""
    _patch_model_ok(monkeypatch)
    _patch_fetch_ok(monkeypatch)
    monkeypatch.setattr(prediction_service, "_execute_async", AsyncMock())

    result = await prediction_service.predict(
        model=MODEL_ID,
        image_url="https://cdn.example.com/a.png",
        user_id=USER_A,
    )

    log = await pred_log_repository.get_by_id(db, result["taskId"])
    assert log is not None
    assert log.recommended_by is None


# ===== C2 推荐报表采纳口径 =====


async def _create_recommendation(db, md5: str, feedback: int = 0) -> int:
    rec = SysRecommendation(
        user_id=USER_A,
        image_md5=md5,
        target_type="model",
        top_models=[{"modelId": MODEL_ID, "modelName": "AECR-NET", "matchScore": 60}],
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
            model_id=MODEL_ID,
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
        model_id=MODEL_ID,
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

    async def ok_model(model_id):
        return SimpleNamespace(model_id=model_id, display_name="AECR-NET")

    monkeypatch.setattr(prediction_service, "get_model", ok_model)
    monkeypatch.setattr(compare_service, "_generate_async", AsyncMock())

    result = await compare_service.generate_report(pred_log, USER_A)

    report_log = await eval_log_repository.get_by_id(db, result["taskId"])
    assert report_log is not None
    assert report_log.task_type == "report"


async def test_metrics_excludes_report_rows(db):
    """/evaluation/metrics 仅返回 task_type=evaluation 行，报告行不混入"""
    eval_log = SysEvalLog(
        model_id=MODEL_ID,
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
        model_id=MODEL_ID,
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
