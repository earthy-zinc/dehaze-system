"""效果评估服务测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖：评估日志用户隔离（list_logs/list_completed_metrics）、合格判定阈值口径、
评估前置失败（算法不存在/图片下载失败）配额归还。

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.entity.sys_log import SysEvalLog
from app.models.enum.log_status import LogStatus
from app.repository.pred_eval_log_repository import eval_log_repository
from app.service.evaluation_service import (
    _is_qualified,
    evaluation_service,
    parse_eval_result,
)
from app.service.member.quota_service import member_quota_service

pytestmark = pytest.mark.requires_db

USER_A = 2001001
USER_B = 2001002


async def _create_log(
    db, user_id: int, algorithm_id: int = 1, status: int = LogStatus.COMPLETED.value
) -> SysEvalLog:
    log = SysEvalLog(
        algorithm_id=algorithm_id,
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


async def test_list_logs_filter_by_algorithm(db):
    """评估日志列表支持按算法 ID 筛选"""
    own_1 = await _create_log(db, USER_A, algorithm_id=1)
    await _create_log(db, USER_A, algorithm_id=2)

    logs, total = await evaluation_service.list_logs(
        db, user_id=USER_A, algorithm_id=1, page=1, size=10
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


# ===== 评估前置失败配额归还 =====


@pytest.fixture
def quota_mocks(monkeypatch):
    deduct = AsyncMock()
    restore = AsyncMock()
    monkeypatch.setattr(member_quota_service, "check_and_deduct_quota", deduct)
    monkeypatch.setattr(member_quota_service, "restore_quota", restore)
    return deduct, restore


async def test_evaluate_quota_restored_when_algorithm_missing(db, quota_mocks, monkeypatch):
    """扣减配额后算法不存在 → 配额归还，不静默泄漏"""
    deduct, restore = quota_mocks

    from app.core.exceptions import BusinessException

    async def raise_missing(algorithm_id):
        raise BusinessException("A0401", "算法不存在")

    monkeypatch.setattr(evaluation_service, "_execute_async", AsyncMock())

    import app.service.prediction.prediction_service as prediction_module

    monkeypatch.setattr(prediction_module.prediction_service, "get_algorithm", raise_missing)

    with pytest.raises(BusinessException):
        await evaluation_service.evaluate(
            algorithm_id=99999999,
            pred_url="http://storage/pred.jpg",
            gt_url="http://storage/gt.jpg",
            user_id=USER_A,
        )

    deduct.assert_awaited_once()
    restore.assert_awaited_once()


async def test_evaluate_quota_restored_when_image_download_fails(db, quota_mocks, monkeypatch):
    """扣减配额后图片下载失败 → 配额归还"""
    deduct, restore = quota_mocks

    import app.service.evaluation_service as evaluation_module
    import app.service.prediction.prediction_service as prediction_module

    async def ok_algorithm(algorithm_id):
        return SimpleNamespace(name="测试算法")

    async def fail_fetch(url):
        raise RuntimeError("图片下载失败")

    monkeypatch.setattr(prediction_module.prediction_service, "get_algorithm", ok_algorithm)
    monkeypatch.setattr(evaluation_module, "fetch_image", fail_fetch)

    with pytest.raises(RuntimeError):
        await evaluation_service.evaluate(
            algorithm_id=1,
            pred_url="http://storage/pred.jpg",
            gt_url="http://storage/gt.jpg",
            user_id=USER_A,
        )

    deduct.assert_awaited_once()
    restore.assert_awaited_once()
