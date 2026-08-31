"""渠道对账域：每日对账任务，比对系统支付流水与渠道账单并记录差异（需求规格 §5.1）。

对账口径：以渠道支付流水号（payment_no）逐单核对金额；系统多单（system_only）、
渠道多单（channel_only）、金额不符（amount_mismatch）均落差异表，由运营跟进处理。
渠道账单下载能力未对接（Mock/未启用渠道返回 None）时跳过该渠道，待渠道真实接入。
"""

import logging
from datetime import datetime, time, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_reconciliation import SysReconciliation
from app.repository.payment_record_repository import payment_record_repository
from app.repository.reconciliation_repository import reconciliation_repository
from app.service.payment_channel_service import payment_channel_service

logger = logging.getLogger(__name__)

DIFF_AMOUNT_MISMATCH = "amount_mismatch"
DIFF_SYSTEM_ONLY = "system_only"
DIFF_CHANNEL_ONLY = "channel_only"


class ReconciliationService:
    def __init__(
        self,
        reconciliation_repository=reconciliation_repository,
        payment_record_repository=payment_record_repository,
        payment_channel_service=payment_channel_service,
    ):
        self.reconciliation_repository = reconciliation_repository
        self.payment_record_repository = payment_record_repository
        self.payment_channel_service = payment_channel_service

    async def run_daily_reconciliation(self, db: AsyncSession, recon_date) -> int:
        """对账指定日期（date），逐渠道比对并全量重写当日差异记录，返回差异数。"""
        start = datetime.combine(recon_date, time.min)
        end = start + timedelta(days=1)

        records = await self.payment_record_repository.list_success_between(db, start, end)
        by_channel: dict[str, list] = {}
        for record in records:
            by_channel.setdefault(record.channel, []).append(record)

        diffs: list[SysReconciliation] = []
        for channel, channel_records in by_channel.items():
            bill_rows = await self.payment_channel_service.download_bill(channel, recon_date)
            if bill_rows is None:
                logger.warning("渠道账单能力未对接，跳过对账 channel=%s date=%s", channel, recon_date)
                continue

            bill_by_no = {row["paymentNo"]: row for row in bill_rows}
            sys_by_no = {r.payment_no: r for r in channel_records}

            for payment_no, record in sys_by_no.items():
                bill = bill_by_no.get(payment_no)
                if bill is None:
                    diffs.append(self._build_diff(recon_date, channel, payment_no, record, DIFF_SYSTEM_ONLY))
                elif int(bill["amount"]) != int(record.amount):
                    diff = self._build_diff(recon_date, channel, payment_no, record, DIFF_AMOUNT_MISMATCH)
                    diff.channel_amount = int(bill["amount"])
                    diffs.append(diff)

            for payment_no, bill in bill_by_no.items():
                if payment_no not in sys_by_no:
                    diffs.append(
                        SysReconciliation(
                            recon_date=recon_date,
                            channel=channel,
                            flow_no=payment_no,
                            order_no=bill.get("orderNo"),
                            channel_amount=int(bill["amount"]),
                            diff_type=DIFF_CHANNEL_ONLY,
                            status=0,
                        )
                    )

        await self.reconciliation_repository.delete_by_date(db, recon_date)
        for diff in diffs:
            await self.reconciliation_repository.create(db, diff)
        await db.flush()

        if diffs:
            logger.warning("渠道对账发现差异 date=%s count=%s", recon_date, len(diffs))
        return len(diffs)

    @staticmethod
    def _build_diff(recon_date, channel: str, payment_no: str, record, diff_type: str) -> SysReconciliation:
        return SysReconciliation(
            recon_date=recon_date,
            channel=channel,
            flow_no=payment_no,
            system_amount=int(record.amount),
            diff_type=diff_type,
            status=0,
        )


reconciliation_service = ReconciliationService()
