"""效果评估服务测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖：评估日志用户隔离（list_logs/list_completed_metrics）、合格判定阈值口径、
远端指标计算落库口径（dehaze-modal /v1/metrics）、评估前置与执行失败配额归还。

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from app.config import settings
from app.models.entity.sys_log import SysEvalLog
from app.models.enum.log_status import LogStatus
from app.repository.pred_eval_log_repository import eval_log_repository
from app.service.evaluation_service import (
    _is_qualified,
    evaluation_service,
    parse_eval_result,
)

pytestmark = pytest.mark.requires_db

USER_A = 2001001
USER_B = 2001002

_MODAL_BASE = settings.MODAL_ENGINE_BASE_URL.rstrip("/")
_MODELS_URL = f"{_MODAL_BASE}/v1/models"
_METRICS_URL = f"{_MODAL_BASE}/v1/metrics"

# 能力发现返回的指标集目录（指标名 + 展示标签 + 是否需参考图）
_METRIC_CATALOG = [
    {
        "id": "psnr",
        "type": "metrics",
        "available": True,
        "capabilities": {"label": "PSNR", "better": "higher", "requires_clear": True},
    },
    {
        "id": "niqe",
        "type": "metrics",
        "available": True,
        "capabilities": {"label": "NIQE", "better": "lower", "requires_clear": False},
    },
]


async def _create_log(
    db, user_id: int, model_id: str = "aecrnet", status: int = LogStatus.COMPLETED.value
) -> SysEvalLog:
    log = SysEvalLog(
        model_id=model_id,
        pred_md5="a" * 32,
        pred_url="http://storage/pred.jpg",
        gt_md5="b" * 32,
        gt_url="http://storage/gt.jpg",
        time=1,
        status=status,
        create_by=user_id,
    )
    return await eval_log_repository.create(db, log)


# ===== 评估日志用户隔离 =====


async def test_list_logs_only_returns_own_logs(db):
    """评估日志列表按用户隔离：不返回其他用户的评估日志"""
    own = await _create_log(db, USER_A)
    await _create_log(db, USER_B)

    logs, total = await evaluation_service.list_logs(db, user_id=USER_A, page=1, size=10)

    assert total == 1
    assert [log.id for log in logs] == [own.id]


async def test_list_logs_filter_by_model(db):
    """评估日志列表支持按模型标识筛选"""
    own_1 = await _create_log(db, USER_A, model_id="aecrnet")
    await _create_log(db, USER_A, model_id="aodnet")

    logs, total = await evaluation_service.list_logs(
        db, user_id=USER_A, model_id="aecrnet", page=1, size=10
    )

    assert total == 1
    assert [log.id for log in logs] == [own_1.id]


async def test_list_completed_metrics_only_completed_and_own(db):
    """评估指标历史：仅返回当前用户已完成记录，处理中/失败不返回"""
    completed = await _create_log(db, USER_A, status=LogStatus.COMPLETED.value)
    await _create_log(db, USER_A, status=LogStatus.PROCESSING.value)
    await _create_log(db, USER_A, status=LogStatus.FAILED.value)
    await _create_log(db, USER_B, status=LogStatus.COMPLETED.value)

    logs, total = await evaluation_service.list_completed_metrics(
        db, user_id=USER_A, page=1, size=10
    )

    assert total == 1
    assert [log.id for log in logs] == [completed.id]


# ===== 合格判定阈值口径 =====


def test_is_qualified_with_label_keys_above_thresholds():
    """指标以 label 为 key（如 "PSNR"）时判定生效：全部达标 → 合格"""
    metrics = {"PSNR": 35.0, "SSIM": 0.9, "LPIPS": 0.2, "NIQE": 4.0}
    assert _is_qualified(metrics) is True


def test_is_qualified_below_any_threshold_fails():
    """任一指标未达阈值 → 不合格"""
    assert _is_qualified({"PSNR": 10.0, "SSIM": 0.9, "LPIPS": 0.2, "NIQE": 4.0}) is False
    assert _is_qualified({"PSNR": 35.0, "SSIM": 0.9, "LPIPS": 0.5, "NIQE": 4.0}) is False
    assert _is_qualified({"PSNR": 35.0, "SSIM": 0.9, "LPIPS": 0.2, "NIQE": 8.0}) is False


def test_is_qualified_empty_metrics_fails():
    """无指标数据 → 不合格"""
    assert _is_qualified({}) is False


# ===== result 归一化（存量 JSON 字符串行 / 双重编码兼容） =====


def test_parse_eval_result_dict_passthrough():
    """JSON 列正常读取（dict）→ 原样返回"""
    assert parse_eval_result({"PSNR": 30.1}) == {"PSNR": 30.1}


def test_parse_eval_result_json_string_parsed():
    """存量行 JSON 字符串（双重编码写入）→ 解析为 dict"""
    assert parse_eval_result('{"PSNR": 12.78, "SSIM": 0.85}') == {
        "PSNR": 12.78,
        "SSIM": 0.85,
    }


def test_parse_eval_result_report_wrapper_parsed():
    """报告任务 result（{reportHtml, generatedAt} 包装对象，存量字符串形式）→ 解析为 dict"""
    raw = '{"reportHtml": "<html/>", "generatedAt": "2026-09-12 00:00:00"}'
    parsed = parse_eval_result(raw)
    assert parsed == {"reportHtml": "<html/>", "generatedAt": "2026-09-12 00:00:00"}


def test_parse_eval_result_invalid_string_returns_none():
    """非合法 JSON 字符串 → 按空处理（不抛 500）"""
    assert parse_eval_result("not-json") is None
    assert parse_eval_result("") is None
    assert parse_eval_result(None) is None
    assert parse_eval_result(123) is None


async def test_update_result_roundtrip_stays_dict(db):
    """回归：update_result 写入 JSON 列后读回必须是 dict（防双重编码复发）"""
    log = await _create_log(db, USER_A, status=LogStatus.PROCESSING.value)
    metrics = {"PSNR": 30.1, "SSIM": 0.9}

    await eval_log_repository.update_result(db, log_id=log.id, result=metrics, time_ms=1500)
    reloaded = await eval_log_repository.get_by_id(db, log.id)
    assert reloaded is not None

    assert reloaded.result == metrics
    assert isinstance(reloaded.result, dict)


# ===== 评估前置校验失败 =====


async def test_evaluate_raises_when_model_missing(db, monkeypatch):
    """模型不存在时提交失败，不创建日志"""
    from app.core.exceptions import BusinessException

    async def raise_missing(model_id):
        raise BusinessException("A0401", "图像模型不存在")

    monkeypatch.setattr(evaluation_service, "_execute_async", AsyncMock())

    import app.service.prediction.prediction_service as prediction_module

    monkeypatch.setattr(prediction_module.prediction_service, "get_model", raise_missing)

    with pytest.raises(BusinessException):
        await evaluation_service.evaluate(
            model_id="no-such-model",
            pred_url="http://storage/pred.jpg",
            gt_url="http://storage/gt.jpg",
            user_id=USER_A,
        )


async def test_evaluate_raises_when_image_download_fails(db, monkeypatch):
    """图片下载失败时提交失败，不创建日志"""
    import app.service.evaluation_service as evaluation_module
    import app.service.prediction.prediction_service as prediction_module

    async def ok_model(model_id):
        return SimpleNamespace(name="测试模型")

    async def fail_fetch(url):
        raise RuntimeError("图片下载失败")

    monkeypatch.setattr(prediction_module.prediction_service, "get_model", ok_model)
    monkeypatch.setattr(evaluation_module, "fetch_image", fail_fetch)

    with pytest.raises(RuntimeError):
        await evaluation_service.evaluate(
            model_id="aecrnet",
            pred_url="http://storage/pred.jpg",
            gt_url="http://storage/gt.jpg",
            user_id=USER_A,
        )


# ===== 远端指标计算（dehaze-modal /v1/metrics） =====


async def _run_evaluation(log_id: int, monkeypatch) -> None:
    """执行异步评估：引擎 HTTP 由 respx 接管，成长值激励不参与本组断言"""

    async def _skip_growth(user_id, log_id):
        return None

    monkeypatch.setattr(evaluation_service, "_award_evaluate_growth", _skip_growth)
    await evaluation_service._execute_async(
        log_id=log_id,
        model_id="aecrnet",
        pred_bytes=BytesIO(b"pred-image"),
        gt_bytes=BytesIO(b"gt-image"),
        user_id=USER_A,
    )


async def test_execute_async_stores_metrics_keyed_by_label(db, monkeypatch):
    """部分指标失败仍落已完成项，且落库 key 为 label（前端按 PSNR/NIQE 取值）"""
    log = await _create_log(db, USER_A, status=LogStatus.PROCESSING.value)

    with respx.mock(assert_all_mocked=True) as router:
        router.get(_MODELS_URL).mock(
            return_value=httpx.Response(200, json={"data": _METRIC_CATALOG})
        )
        route = router.post(_METRICS_URL).mock(
            return_value=httpx.Response(
                200, json={"metrics": {"psnr": 32.5}, "failed": ["niqe"], "meta": {}}
            )
        )
        await _run_evaluation(log.id, monkeypatch)

    # 下发的指标名清单来自能力发现（全量指标，含逐项失败项）
    body = route.calls.last.request.read().decode("utf-8", "replace")
    assert '"psnr"' in body
    assert '"niqe"' in body

    await db.refresh(log)
    assert log.status == LogStatus.COMPLETED.value
    assert log.result == {"PSNR": 32.5}


async def test_execute_async_fails_when_all_metrics_fail(db, monkeypatch):
    """全部指标计算失败 → 任务置失败（部分失败才算已完成；评估免费不涉及配额）"""
    log = await _create_log(db, USER_A, status=LogStatus.PROCESSING.value)

    with respx.mock(assert_all_mocked=True) as router:
        router.get(_MODELS_URL).mock(
            return_value=httpx.Response(200, json={"data": _METRIC_CATALOG})
        )
        router.post(_METRICS_URL).mock(
            return_value=httpx.Response(
                200, json={"metrics": {}, "failed": ["psnr", "niqe"], "meta": {}}
            )
        )
        await _run_evaluation(log.id, monkeypatch)

    await db.refresh(log)
    assert log.status == LogStatus.FAILED.value
    assert "全部失败" in (log.error_message or "")
