"""渠道对账服务测试：金额不符/系统多单/渠道多单三类差异、账单能力未对接跳过、重跑全量重写。"""

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.entity.sys_payment_record import SysPaymentRecord
from app.repository.payment_record_repository import payment_record_repository
from app.repository.reconciliation_repository import reconciliation_repository
from app.service.order.reconciliation_service import ReconciliationService

pytestmark = pytest.mark.requires_db

RECON_DATE = date(2026, 8, 28)


def _channel_service(bill_rows):
    return SimpleNamespace(download_bill=AsyncMock(return_value=bill_rows))


async def _seed_payment(db, payment_no: str, amount: int, channel: str = "wechat"):
    record = SysPaymentRecord(
        order_id=1,
        user_id=100,
        payment_no=payment_no,
        channel=channel,
        amount=amount,
        status=2,
        callback_time=datetime.combine(RECON_DATE, datetime.min.time()) + timedelta(hours=12),
    )
    await payment_record_repository.create(db, record)
    await db.flush()
    return record


def _bill(payment_no: str, amount: int, order_no: str | None = None):
    row = {"paymentNo": payment_no, "amount": amount}
    if order_no:
        row["orderNo"] = order_no
    return row


class TestRunDailyReconciliation:
    async def test_amount_mismatch_recorded(self, db):
        await _seed_payment(db, "PAY-A", 5000)
        svc = ReconciliationService(
            payment_channel_service=_channel_service([_bill("PAY-A", 4000)])
        )
        count = await svc.run_daily_reconciliation(db, RECON_DATE)
        assert count == 1

        diffs = await reconciliation_repository.list_by_date(db, RECON_DATE)
        assert diffs[0].diff_type == "amount_mismatch"
        assert diffs[0].system_amount == 5000
        assert diffs[0].channel_amount == 4000

    async def test_system_only_and_channel_only(self, db):
        await _seed_payment(db, "PAY-SYS", 5000)
        svc = ReconciliationService(
            payment_channel_service=_channel_service([_bill("PAY-CH", 3000, order_no="DH1")])
        )
        count = await svc.run_daily_reconciliation(db, RECON_DATE)
        assert count == 2

        diffs = {d.flow_no: d for d in await reconciliation_repository.list_by_date(db, RECON_DATE)}
        assert diffs["PAY-SYS"].diff_type == "system_only"
        assert diffs["PAY-SYS"].channel_amount is None
        assert diffs["PAY-CH"].diff_type == "channel_only"
        assert diffs["PAY-CH"].system_amount is None
        assert diffs["PAY-CH"].order_no == "DH1"

    async def test_matched_records_produce_no_diff(self, db):
        await _seed_payment(db, "PAY-OK", 5000)
        svc = ReconciliationService(
            payment_channel_service=_channel_service([_bill("PAY-OK", 5000)])
        )
        count = await svc.run_daily_reconciliation(db, RECON_DATE)
        assert count == 0
        assert await reconciliation_repository.list_by_date(db, RECON_DATE) == []

    async def test_bill_unavailable_skips_channel(self, db):
        # Mock/未启用渠道无账单能力（download_bill 返回 None）→ 跳过，不产生差异
        await _seed_payment(db, "PAY-SKIP", 5000)
        svc = ReconciliationService(
            payment_channel_service=SimpleNamespace(download_bill=AsyncMock(return_value=None))
        )
        count = await svc.run_daily_reconciliation(db, RECON_DATE)
        assert count == 0

    async def test_rerun_replaces_same_date_diffs(self, db):
        await _seed_payment(db, "PAY-R1", 5000)
        svc = ReconciliationService(
            payment_channel_service=_channel_service([_bill("PAY-R1", 1000)])
        )
        await svc.run_daily_reconciliation(db, RECON_DATE)

        # 第二次运行账单一致 → 旧差异被清空
        svc_ok = ReconciliationService(
            payment_channel_service=_channel_service([_bill("PAY-R1", 5000)])
        )
        await svc_ok.run_daily_reconciliation(db, RECON_DATE)
        assert await reconciliation_repository.list_by_date(db, RECON_DATE) == []
