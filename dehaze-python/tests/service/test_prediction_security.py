"""去雾处理模块安全与状态机回归（F-REC-002 审查修复）。

覆盖审查发现：任务归属越权（查询/取消/日志列表）、取消后结果复活竞态、
推理超时、配额扣减顺序、批量上限权威来源、对抗性语料、SSRF/本地路径防护。
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_log import SysPredLog
from app.models.enum.log_status import LogStatus
from app.repository.pred_eval_log_repository import pred_log_repository
from app.service.prediction import inference_executor
from app.service.prediction.image_source import _assert_public_host, fetch_image
from app.service.prediction.inference_executor import run_dehaze
from app.service.prediction.prediction_service import prediction_service

pytestmark = pytest.mark.requires_db


async def _create_log(
    db, create_by: int | None, status: int = LogStatus.PROCESSING.value
) -> SysPredLog:
    log = SysPredLog(
        algorithm_id=1,
        origin_md5="0" * 32,
        origin_url="http://example.com/a.png",
        pred_md5="",
        pred_url="",
        time=0,
        status=status,
        create_by=create_by,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


# ===== 归属校验（越权修复） =====


async def test_cancel_other_users_task_forbidden(db):
    """用户 B 取消用户 A 的任务 → A0401（防枚举，与不存在同口径）"""
    log = await _create_log(db, create_by=101)
    with pytest.raises(BusinessException) as ei:
        await prediction_service.cancel_task(db, log.id, user_id=202)
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_cancel_own_task_and_idempotent(db):
    """本人取消成功且幂等：二次取消返回当前状态，不重复流转"""
    log = await _create_log(db, create_by=101)
    result = await prediction_service.cancel_task(db, log.id, user_id=101)
    assert result["status"] == LogStatus.CANCELLED.value

    result2 = await prediction_service.cancel_task(db, log.id, user_id=101)
    assert result2["status"] == LogStatus.CANCELLED.value

    await db.refresh(log)
    assert log.status == LogStatus.CANCELLED.value


async def test_cancelled_owner_null_task_forbidden(db):
    """create_by 为空（系统写入）的任务不可被任何用户取消"""
    log = await _create_log(db, create_by=None)
    with pytest.raises(BusinessException) as ei:
        await prediction_service.cancel_task(db, log.id, user_id=101)
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_list_logs_scoped_to_owner(db):
    """日志列表仅返回本人记录"""
    await _create_log(db, create_by=101)
    await _create_log(db, create_by=202)
    logs, total = await prediction_service.list_logs(db, user_id=101, page=1, size=10)
    assert total == 1
    assert all(log.create_by == 101 for log in logs)


# ===== 状态机 guard（取消后复活竞态） =====


async def test_cancelled_task_not_resurrected_by_inference_result(db):
    """取消后后台推理完成：update_result 不得把 CANCELLED 覆盖为 COMPLETED"""
    log = await _create_log(db, create_by=101)
    await pred_log_repository.update_status(db, log.id, LogStatus.CANCELLED.value, "任务已取消", 0)
    updated = await pred_log_repository.update_result(
        db, log.id, pred_md5="a" * 32, pred_url="http://x/r.png", time_ms=100
    )
    assert updated is False
    await db.refresh(log)
    assert log.status == LogStatus.CANCELLED.value
    assert log.pred_url == ""


async def test_terminal_state_not_overwritten(db):
    """终态（failed）不可再被流转覆盖"""
    log = await _create_log(db, create_by=101)
    ok = await pred_log_repository.update_status(
        db, log.id, LogStatus.FAILED.value, "算法执行失败", 5
    )
    assert ok is True

    again = await pred_log_repository.update_status(
        db, log.id, LogStatus.FAILED.value, "算法执行失败", 5
    )
    assert again is False

    resurrect = await pred_log_repository.update_result(
        db, log.id, pred_md5="a" * 32, pred_url="http://x/r.png", time_ms=100
    )
    assert resurrect is False


# ===== 推理超时（30s 阈值） =====


async def test_inference_timeout_raises_business_error(monkeypatch):
    """推理超过阈值 → BusinessException（超时口径），任务按失败处理并回滚配额"""

    def _slow_sync(import_path, model_path, image_bytes):
        import time

        time.sleep(2)

    monkeypatch.setattr(inference_executor, "INFERENCE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(inference_executor, "_run_dehaze_sync", _slow_sync)

    with pytest.raises(BusinessException) as ei:
        await run_dehaze("algorithm.DCP.run", "", __import__("io").BytesIO(b"x"))
    assert ei.value.code == ResultCode.SYSTEM_EXECUTION_ERROR
    assert "超时" in ei.value.message


# ===== 对抗性语料：损坏图片走失败路径（不崩、不返回伪结果） =====


async def test_corrupted_image_raises_clean_business_error():
    """随机字节作为图片 → 算法执行失败（包装为业务错误，截断内部细节）"""
    import io as _io

    garbage = _io.BytesIO(b"\x00\xff\xfe" * 512)
    with pytest.raises(BusinessException) as ei:
        await run_dehaze("algorithm.DCP.run", "", garbage)
    assert ei.value.code == ResultCode.SYSTEM_EXECUTION_ERROR


# ===== fetch_image：本地路径 / SSRF 防护 =====


async def test_fetch_image_rejects_local_path():
    """imageUrl 本地路径分支已移除（任意文件读取漏洞）"""
    with pytest.raises(BusinessException) as ei:
        await fetch_image("/etc/passwd")
    assert ei.value.code == ResultCode.PARAM_ERROR


async def test_fetch_image_rejects_non_http_scheme():
    with pytest.raises(BusinessException) as ei:
        await fetch_image("ftp://example.com/a.png")
    assert ei.value.code == ResultCode.PARAM_ERROR


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9000/bucket/a.png",
        "http://localhost/a.png",
        "http://10.0.0.5/a.png",
        "http://192.168.1.10/a.png",
        "http://172.16.0.9/a.png",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/a.png",
        "http://0.0.0.0/a.png",
    ],
)
async def test_assert_public_host_blocks_internal_targets(url):
    """SSRF 防护：回环/私网/链路本地/未指定地址一律拒绝"""
    with pytest.raises(BusinessException):
        await _assert_public_host(url)


# ===== predict：算法不存在不扣配额（扣减顺序修复） =====


async def test_predict_missing_algorithm_skips_quota_deduct(db):
    from app.service.member import quota_service as quota_module

    deduct = AsyncMock()
    with (
        patch.object(quota_module.member_quota_service, "check_and_deduct_quota", deduct),
        pytest.raises(BusinessException) as ei,
    ):
        await prediction_service.predict(
            algorithm_id=99999999,
            image_url="http://example.com/a.png",
            user_id=101,
        )
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND
    deduct.assert_not_awaited()


# ===== 批量上限：benefit.batch_limit 为唯一权威（去除 20 张硬编码） =====


def _batch_items(n: int):
    from app.models.schema.prediction import BatchPredictionItem

    return [BatchPredictionItem(fileId=i) for i in range(1, n + 1)]


async def test_batch_beyond_20_allowed_for_high_tier(db):
    """高等级权益 batch_limit=1000 时提交 21 张不再被硬编码拦截"""
    from app.repository.member_benefit_repository import member_benefit_repository
    from app.repository.member_repository import member_repository
    from tests.stubs.factories import make_benefit, make_member

    member = make_member(level_code="level_3")
    benefit = make_benefit(batch_limit=1000)
    predict_mock = AsyncMock(return_value={"logId": 1, "status": 1})

    with (
        patch.object(member_repository, "get_by_user_id", AsyncMock(return_value=member)),
        patch.object(
            member_benefit_repository, "get_by_level_code", AsyncMock(return_value=benefit)
        ),
        patch.object(prediction_service, "predict", predict_mock),
    ):
        results = await prediction_service.batch_predict(
            1, _batch_items(21), user_id=101, skip_quota_check=False
        )
    assert len(results) == 21
    assert predict_mock.await_count == 21
