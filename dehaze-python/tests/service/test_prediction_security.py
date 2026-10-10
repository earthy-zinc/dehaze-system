"""图像任务安全与状态机回归（统一模型口径）。

覆盖：任务归属越权（查询/取消/列表）、取消后结果复活竞态、终态不可覆盖、
模型路由拒绝（不存在/停用/类型不符）、输入图片来源防护（本地路径 / 非 http / SSRF）。
"""

from unittest.mock import AsyncMock

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_log import SysPredLog
from app.models.enum.log_status import LogStatus
from app.repository.pred_eval_log_repository import pred_log_repository
from app.service.prediction import prediction_service as prediction_module
from app.service.prediction.image_source import _assert_public_host, fetch_image
from app.service.prediction.prediction_service import prediction_service

pytestmark = pytest.mark.requires_db

# 种子图像模型：aecrnet（启用）/ daclipuir（停用）/ qwen3-0.6b（chat，类型不符）
MODEL_ENABLED = "aecrnet"
MODEL_DISABLED = "daclipuir"
MODEL_WRONG_TYPE = "qwen3-0.6b"


async def _create_log(
    db, create_by: int | None, status: int = LogStatus.PROCESSING.value
) -> SysPredLog:
    log = SysPredLog(
        model_id=MODEL_ENABLED,
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
        db, log.id, LogStatus.FAILED.value, "推理服务失败", 5
    )
    assert ok is True

    again = await pred_log_repository.update_status(
        db, log.id, LogStatus.FAILED.value, "推理服务失败", 5
    )
    assert again is False

    resurrect = await pred_log_repository.update_result(
        db, log.id, pred_md5="a" * 32, pred_url="http://x/r.png", time_ms=100
    )
    assert resurrect is False


# ===== 模型路由：不可用即拒绝，且不触发计费预校验 =====


@pytest.mark.parametrize("model", [MODEL_DISABLED, MODEL_WRONG_TYPE, "no-such-model"])
async def test_predict_rejects_unavailable_image_model(db, model, monkeypatch):
    """停用 / 类型不符 / 不存在的模型 → A0400，且在扣费预校验之前拒绝"""
    ensure_balance = AsyncMock()
    monkeypatch.setattr(prediction_module.image_billing_service, "ensure_balance", ensure_balance)

    with pytest.raises(BusinessException) as ei:
        await prediction_service.predict(
            model=model, image_url="https://cdn.example.com/a.png", user_id=101
        )
    assert ei.value.code == ResultCode.PARAM_ERROR
    ensure_balance.assert_not_awaited()


async def test_predict_missing_file_raises_not_found(db, monkeypatch):
    """fileId 指向不存在的文件 → A0401"""
    monkeypatch.setattr(prediction_module.image_billing_service, "ensure_balance", AsyncMock())
    with pytest.raises(BusinessException) as ei:
        await prediction_service.predict(
            model=MODEL_ENABLED, image_url="", user_id=101, file_id=99999999
        )
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


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
