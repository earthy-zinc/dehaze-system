"""图像链路核心用例（统一模型口径，阶段 3）。

分三组：
- 模型路由：请求 model 优先 > 平台默认（字典 ai_model.default_model.image_edit）> 拒绝
- 按张计费：成功计费、缓存命中不计费、失败/取消不计费、无价格计 0 且 error_code 为 NULL
- WPXNet 前缀拦截：仅 `wpxnet-` 前缀模型触发预查询短路
"""

import hashlib
from decimal import Decimal
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_credit_log import SysAiCreditLog
from app.models.entity.sys_ai_model import SysAiModel
from app.models.entity.sys_file import SysFile
from app.models.entity.sys_log import SysPredLog
from app.models.entity.sys_wpx_file import SysWpxFile
from app.models.enum.log_status import LogStatus
from app.repository.ai_billing_repository import AiBillingRepository
from app.repository.ai_credit_log_repository import AiCreditLogRepository
from app.repository.pred_eval_log_repository import pred_log_repository
from app.service.billing.balance_service import BalanceService
from app.service.billing.quota_service import QuotaService
from app.service.prediction import prediction_service as prediction_module
from app.service.prediction.image_billing_service import ImageBillingService
from app.service.prediction.image_route import resolve_image_model
from app.service.prediction.interceptor import PredictionContext
from app.service.prediction.prediction_service import prediction_service
from app.service.prediction.wpxnet_interceptor import WpxNetPredictionInterceptor

pytestmark = pytest.mark.requires_db

MODEL_ENABLED = "aecrnet"
MODEL_DISABLED = "daclipuir"
MODEL_WRONG_TYPE = "qwen3-0.6b"
MODEL_WPXNET = "wpxnet-i-haze"
PROVIDER_MODEI_ID = 79


# ===== 模型路由 =====


async def test_route_uses_explicit_model(db):
    route = await resolve_image_model(db, MODEL_ENABLED)
    assert route.model_id == MODEL_ENABLED
    assert route.model.model_type == "image_edit"
    assert route.provider.id == PROVIDER_MODEI_ID


async def test_route_falls_back_to_platform_default(db):
    """请求 model 为空 → 取字典 ai_model.default_model.image_edit（种子为 aecrnet）"""
    route = await resolve_image_model(db, None)
    assert route.model_id == MODEL_ENABLED


async def test_route_rejects_blank_request_with_empty_default(db, mock_redis):
    """平台默认缺项且请求为空 → A0500（业务错误，不静默回退代码常量）"""
    from app.repository.dict_repository import dict_repository
    from app.service.dict_service import _invalidate_dict_value_cache

    item = await dict_repository.get_by_type_code_and_name(
        db, "ai_model", "default_model.image_edit"
    )
    assert item is not None
    item.deleted = item.id
    await db.flush()
    await _invalidate_dict_value_cache(mock_redis, "ai_model")

    with pytest.raises(BusinessException) as ei:
        await resolve_image_model(db, None)
    assert ei.value.code == ResultCode.BUSINESS_ERROR


@pytest.mark.parametrize("model", [MODEL_DISABLED, MODEL_WRONG_TYPE, "no-such-model"])
async def test_route_rejects_unavailable_model(db, model):
    with pytest.raises(BusinessException) as ei:
        await resolve_image_model(db, model)
    assert ei.value.code == ResultCode.PARAM_ERROR


# ===== 按张计费 =====


class StubPricing:
    def __init__(self, credits: int, configured: bool = True):
        self._credits = credits
        self._configured = configured

    async def calculate_unit_credits(self, db, model_id, provider_id, at_time, quantity):
        return {"credits": self._credits, "configured": self._configured, "billing_unit": "image"}


class StubQuotaService(QuotaService):
    def __init__(self):
        self.deducted: list[int] = []

    async def deduct(self, user_id, credits):
        self.deducted.append(credits)


class StubBalanceService(BalanceService):
    def __init__(self):
        self.deducted: list[int] = []

    async def deduct(self, db, user_id, credits):
        self.deducted.append(credits)

    async def get_balance(self, db, user_id):
        return Decimal(10000)


class StubBillingRepo(AiBillingRepository):
    def __init__(self):
        self.created: list[dict] = []

    async def create_billing(self, db: AsyncSession, **kwargs) -> SysAiBilling:
        self.created.append(kwargs)
        return SysAiBilling(id=9001 + len(self.created))


class StubCreditLogRepo(AiCreditLogRepository):
    def __init__(self):
        self.logs: list[dict] = []

    async def create_log(self, db: AsyncSession, **kwargs) -> SysAiCreditLog:
        self.logs.append(kwargs)
        return SysAiCreditLog(id=7001 + len(self.logs))


def _billing_service(credits: int, configured: bool = True):
    quota, balance = StubQuotaService(), StubBalanceService()
    billing_repo, credit_repo = StubBillingRepo(), StubCreditLogRepo()
    svc = ImageBillingService(
        ai_billing_repository=billing_repo,
        ai_credit_log_repository=credit_repo,
        balance_service=balance,
        quota_service=quota,
        price_service=StubPricing(credits, configured),
    )
    return svc, quota, balance, billing_repo, credit_repo


async def test_charge_image_records_bill_type_and_quantity(db):
    """成功计费：按张落 sys_ai_billing，bill_type=image / quantity_unit=image"""
    svc, quota, balance, billing_repo, credit_repo = _billing_service(credits=3)

    credits = await svc.charge_image(db, 101, MODEL_ENABLED, PROVIDER_MODEI_ID, 1)

    assert credits == 3
    assert quota.deducted == [3]
    assert balance.deducted == [3]
    row = billing_repo.created[0]
    assert row["bill_type"] == "image"
    assert row["model"] == MODEL_ENABLED
    assert row["quantity"] == Decimal(1)
    assert row["quantity_unit"] == "image"
    assert row["error_code"] is None
    assert credit_repo.logs
    assert credit_repo.logs[0]["amount"] == Decimal(-3)


async def test_charge_image_without_price_records_zero_credits(db):
    """无价格：计 0 分、error_code 保持 NULL、不扣配额余额、不记流水（不阻断调用）"""
    svc, quota, balance, billing_repo, credit_repo = _billing_service(credits=0, configured=False)

    credits = await svc.charge_image(db, 101, MODEL_ENABLED, PROVIDER_MODEI_ID, 1)

    assert credits == 0
    assert quota.deducted == []
    assert balance.deducted == []
    row = billing_repo.created[0]
    assert row["credits"] == 0
    assert row["error_code"] is None  # 正向：成功计费行 error_code 为 NULL（负向不得出现 no_price）
    assert credit_repo.logs == []


async def _create_pending_log(db) -> SysPredLog:
    log = SysPredLog(
        model_id=MODEL_ENABLED,
        origin_md5="0" * 32,
        origin_url="https://cdn.example.com/a.png",
        pred_md5="",
        pred_url="",
        time=0,
        status=LogStatus.PROCESSING.value,
        create_by=101,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


def _patch_run_edit(
    monkeypatch, *, result: bytes | None = b"png-bytes", error: Exception | None = None
):
    async def _run_edit(route, image_bytes, params):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(prediction_module, "run_edit", _run_edit)
    monkeypatch.setattr(
        prediction_module, "upload_result", AsyncMock(return_value="predictions/20260101/out.png")
    )


async def test_execute_async_charges_once_on_success(db, monkeypatch):
    log = await _create_pending_log(db)
    _patch_run_edit(monkeypatch)
    charge = AsyncMock()
    monkeypatch.setattr(prediction_module.image_billing_service, "charge_image", charge)

    await prediction_service._execute_async(
        log_id=log.id,
        route=await resolve_image_model(db, MODEL_ENABLED),
        image_bytes=BytesIO(b"fake"),
        image_md5="a" * 32,
        cache_key=f"image:{MODEL_ENABLED}:{'a' * 32}",
        user_id=101,
        billed=True,
    )

    charge.assert_awaited_once()
    await db.refresh(log)
    assert log.status == LogStatus.COMPLETED.value


async def test_execute_async_does_not_charge_on_failure(db, monkeypatch):
    log = await _create_pending_log(db)
    _patch_run_edit(monkeypatch, error=RuntimeError("图像引擎服务不可用"))
    charge = AsyncMock()
    monkeypatch.setattr(prediction_module.image_billing_service, "charge_image", charge)

    await prediction_service._execute_async(
        log_id=log.id,
        route=await resolve_image_model(db, MODEL_ENABLED),
        image_bytes=BytesIO(b"fake"),
        image_md5="a" * 32,
        cache_key=f"image:{MODEL_ENABLED}:{'a' * 32}",
        user_id=101,
        billed=True,
    )

    charge.assert_not_awaited()
    await db.refresh(log)
    assert log.status == LogStatus.FAILED.value


async def test_execute_async_does_not_charge_when_cancelled(db, monkeypatch):
    """任务已被取消：结果丢弃，不写缓存、不计费"""
    log = await _create_pending_log(db)
    await pred_log_repository.update_status(db, log.id, LogStatus.CANCELLED.value, "任务已取消", 0)
    _patch_run_edit(monkeypatch)
    charge = AsyncMock()
    monkeypatch.setattr(prediction_module.image_billing_service, "charge_image", charge)

    await prediction_service._execute_async(
        log_id=log.id,
        route=await resolve_image_model(db, MODEL_ENABLED),
        image_bytes=BytesIO(b"fake"),
        image_md5="a" * 32,
        cache_key=f"image:{MODEL_ENABLED}:{'a' * 32}",
        user_id=101,
        billed=True,
    )

    charge.assert_not_awaited()
    await db.refresh(log)
    assert log.status == LogStatus.CANCELLED.value


async def test_cache_hit_does_not_charge(db, mock_redis, monkeypatch):
    """缓存命中直接 completed（走 _write_completed_log），不计费"""
    image_bytes = b"cached-payload"
    image_md5 = hashlib.md5(image_bytes).hexdigest()

    async def _fetch(url):
        return BytesIO(image_bytes)

    monkeypatch.setattr(prediction_module, "fetch_image", _fetch)
    monkeypatch.setattr(prediction_module.image_billing_service, "ensure_balance", AsyncMock())
    charge = AsyncMock()
    monkeypatch.setattr(prediction_module.image_billing_service, "charge_image", charge)
    await mock_redis.set(
        f"image:{MODEL_ENABLED}:{image_md5}",
        '{"storage": "local", "objectName": "predictions/x.png", "resultMd5": "d' + "0" * 31 + '"}',
    )

    result = await prediction_service.predict(
        model=MODEL_ENABLED, image_url="https://cdn.example.com/a.png", user_id=101
    )

    assert result["status"] == LogStatus.COMPLETED.value
    charge.assert_not_awaited()


# ===== WPXNet 前缀拦截 =====


def _wpx_model() -> SysAiModel:
    return SysAiModel(
        id=479,
        provider_id=PROVIDER_MODEI_ID,
        model_id=MODEL_WPXNET,
        model_type="image_edit",
        display_name="I-HAZE",
    )


async def test_wpxnet_interceptor_skips_non_prefix_model(db):
    interceptor = WpxNetPredictionInterceptor()
    ctx = PredictionContext(
        model=SysAiModel(model_id=MODEL_ENABLED, model_type="image_edit", display_name="AECR-NET")
    )
    assert await interceptor.intercept(ctx) is None


async def test_wpxnet_interceptor_miss_without_mapping(db):
    interceptor = WpxNetPredictionInterceptor()
    origin_file = SysFile(
        type="prediction",
        name="a.png",
        object_name="predictions/a.png",
        storage="local",
        size="1",
        size_bytes=1,
        md5="0" * 32,
    )
    ctx = PredictionContext(model=_wpx_model(), origin_file=origin_file)
    assert await interceptor.intercept(ctx) is None


async def test_wpxnet_interceptor_hit_returns_presigned_result(db):
    """前缀命中且存在映射 → 短路返回映射结果（不调用模型）"""
    origin_md5 = "1" * 32
    origin_file = SysFile(
        type="prediction",
        name="a.png",
        object_name="predictions/a.png",
        storage="local",
        size="1",
        size_bytes=1,
        md5=origin_md5,
    )
    db.add(origin_file)
    await db.flush()

    new_file = SysFile(
        type="prediction",
        name="a_dehazed.png",
        object_name="predictions/a_dehazed.png",
        storage="local",
        size="1",
        size_bytes=1,
        md5="2" * 32,
    )
    db.add(new_file)
    await db.flush()
    db.add(
        SysWpxFile(
            origin_file_id=origin_file.id,
            origin_md5=origin_md5,
            origin_path="wpx/origin/a.png",
            new_file_id=new_file.id,
            new_path="wpx/new/a.png",
            new_md5="2" * 32,
        )
    )
    await db.flush()

    interceptor = WpxNetPredictionInterceptor()
    ctx = PredictionContext(model=_wpx_model(), origin_file=origin_file)
    result = await interceptor.intercept(ctx)

    assert result is not None
    assert result.result_md5 == "2" * 32
    assert "predictions/a_dehazed.png" in result.result_url
