"""退款(售后)域：申请/审核（通过/驳回）/列表/失败重试。

退款定位为平台兜底保障而非用户常规权益：原因类型必选，按商品类型折算退款金额，
审核通过后按原支付方式退款并回退履约。
"""

import logging
import math
import random
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_balance_refund import SysBalanceRefund
from app.models.entity.sys_refund_record import SysRefundRecord
from app.repository.balance_refund_repository import balance_refund_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.repository.order_repository import order_repository
from app.repository.payment_record_repository import payment_record_repository
from app.repository.refund_record_repository import refund_record_repository
from app.service.order.order_service import (
    _invalidate_order_detail_cache,
    _refund_to_vo,
)
from app.service.payment_channel_service import payment_channel_service

logger = logging.getLogger(__name__)

REFUND_MAX_RETRY_COUNT = 3

VALID_REASON_TYPES = {"after_sale", "force_majeure", "merchant", "other"}


def _gen_refund_no() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    rand = random.randint(100000, 999999)
    return f"RF{ts}{rand}"


def _restore_status_for_order(order) -> int:
    """退款未成功时订单回退状态：会员卡未到期回已完成(3)，否则回已支付(2)。"""
    now = datetime.now()
    if order.package_expire_time and order.package_expire_time > now:
        return 3
    return 2


def _calc_refund_amount(order, reason_type: str) -> tuple[int, int | None, int | None]:
    """按商品类型折算退款金额，返回 (refund_amount, used_days, used_credits)。"""
    if reason_type == "merchant":
        return order.paid_amount, None, None

    paid_time = order.paid_time or datetime.now()

    if order.package_type == "credit":
        credit_amount = int(order.credit_amount or 0)
        # 已消耗积分由 AI 计费模块提供；当前按未消耗处理（used_credits=0 全额折算）
        used_credits = 0
        if credit_amount <= 0:
            return 0, None, used_credits
        refund_amount = order.paid_amount * (credit_amount - used_credits) // credit_amount
        return refund_amount, None, used_credits

    # vip 按天折算
    period_days = int(getattr(order, "period_days", None) or 0)
    if period_days <= 0:
        return 0, None, None
    used_days = max(1, math.ceil((datetime.now() - paid_time).total_seconds() / 86400))
    remaining_days = period_days - used_days
    if remaining_days <= 0:
        return 0, used_days, None
    refund_amount = order.paid_amount * remaining_days // period_days
    return refund_amount, used_days, None


class RefundService:
    def __init__(
        self,
        mongo_audit_log_repository=mongo_audit_log_repository,
        order_repository=order_repository,
        payment_record_repository=payment_record_repository,
        refund_record_repository=refund_record_repository,
        balance_refund_repository=balance_refund_repository,
        payment_channel_service=payment_channel_service,
        balance_account_service=None,
        member_service=None,
        ai_balance_service=None,
    ):
        self.mongo_audit_log_repository = mongo_audit_log_repository
        self.order_repository = order_repository
        self.payment_record_repository = payment_record_repository
        self.refund_record_repository = refund_record_repository
        self.balance_refund_repository = balance_refund_repository
        self.payment_channel_service = payment_channel_service
        if balance_account_service is None:
            from app.service.order.balance_account_service import (
                balance_account_service as _b,
            )

            balance_account_service = _b
        self.balance_account_service = balance_account_service
        if member_service is None:
            from app.service.member.member_service import member_service as _m

            member_service = _m
        self.member_service = member_service
        if ai_balance_service is None:
            from app.service.billing.balance_service import balance_service as _ab

            ai_balance_service = _ab
        self.ai_balance_service = ai_balance_service

    async def apply_refund(self, db: AsyncSession, order_no: str, form: dict, user_id: int) -> dict:
        order = await self.order_repository.get_by_order_no(db, order_no)
        if not order:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)
        if order.user_id != user_id:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        if order.status not in (2, 3):
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)

        reason_type = form.get("reasonType")
        if reason_type not in VALID_REASON_TYPES:
            raise BusinessException(ResultCode.PARAM_ERROR, "请选择有效的售后原因类型")

        existing = await self.refund_record_repository.get_by_order_id(db, order.id)
        if existing:
            raise BusinessException(ResultCode.REFUND_ALREADY_EXISTS)

        custom_reason = form.get("customReason")
        if custom_reason:
            reason = f"{reason_type}:{custom_reason}"
        else:
            reason = form.get("reason") or reason_type

        refund_amount, used_days, used_credits = _calc_refund_amount(order, reason_type)

        refund = SysRefundRecord(
            refund_no=_gen_refund_no(),
            order_id=order.id,
            user_id=user_id,
            refund_amount=refund_amount,
            reason_type=reason_type,
            reason=reason,
            used_days=used_days,
            used_credits=used_credits,
            status=1,
            channel=order.pay_method,
            apply_time=datetime.now(),
        )
        await self.refund_record_repository.create(db, refund)

        order.status = 5
        await db.flush()
        await _invalidate_order_detail_cache(order_no)

        self.mongo_audit_log_repository.create_audit_async(
            operator_id=user_id,
            target_type="order",
            target_id=order_no,
            action="refund_apply",
            module="order",
            after_value=form if not hasattr(form, "dict") else form,
        )
        return {"refundNo": refund.refund_no, "refundAmount": refund_amount}

    async def apply_balance_refund(
        self, db: AsyncSession, user_id: int, form: dict
    ) -> dict:
        """用户提交余额退款申请（充值余额退回）。

        仅创建申请记录，余额/冻结校验留待管理员审核环节。
        """
        balance = await self.balance_account_service.get_balance(db, user_id)
        amount = form.get("amount")
        if amount is None or amount <= 0:
            amount = balance["balance"]
        record = SysBalanceRefund(
            refund_no=f"BR{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}",
            user_id=user_id,
            amount=amount,
            status=1,
            apply_time=datetime.now(),
        )
        await self.balance_refund_repository.create(db, record)
        return {"refundNo": record.refund_no, "amount": record.amount}

    async def approve_balance_refund(
        self, db: AsyncSession, refund_id: int, form: dict, auditor_id: int
    ) -> None:
        """管理员审核余额退款：校验余额与冻结，原路退回渠道后扣减可用余额。"""
        record = await self.balance_refund_repository.get_by_id(db, refund_id)
        if not record:
            raise BusinessException(ResultCode.REFUND_NOT_FOUND)
        if record.status != 1:
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)

        account = await self.balance_account_service.get_account(db, record.user_id)
        if not account or account.balance < record.amount:
            raise BusinessException(ResultCode.BALANCE_INSUFFICIENT)
        if account.frozen_balance > 0:
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID, "存在冻结余额，暂不可退")

        now = datetime.now()
        record.audit_time = now
        record.auditor_id = auditor_id
        record.audit_remark = form.get("remark", "")

        # 原路退回（渠道未启用时 mock 成功）
        channel_refund_no = None
        try:
            channel = form.get("channel")
            if channel in ("wechat", "alipay"):
                result = await self.payment_channel_service.refund(
                    channel, record.refund_no, "", record.amount, record.amount
                )
                if not result.success:
                    raise BusinessException(
                        ResultCode.SYSTEM_EXECUTION_ERROR,
                        result.error_message or "渠道退款失败",
                    )
                channel_refund_no = result.channel_refund_no
                record.channel = channel
            await self.balance_account_service.withdraw(db, record.user_id, record.amount)
        except Exception as e:
            logger.error("余额退款执行失败 refundId=%s: %s", refund_id, e)
            record.status = 3
            record.error_message = str(e)
        else:
            record.status = 2
            record.refund_time = now
            record.channel_refund_no = channel_refund_no
        await db.flush()

        self.mongo_audit_log_repository.create_audit_async(
            operator_id=auditor_id,
            target_type="balance_refund",
            target_id=refund_id,
            action="balance_refund_approve",
            module="order",
            after_value=form if not hasattr(form, "dict") else form,
        )

    async def approve_refund(
        self, db: AsyncSession, refund_id: int, form: dict, auditor_id: int
    ) -> None:
        refund = await self.refund_record_repository.get_by_id(db, refund_id)
        if not refund:
            raise BusinessException(ResultCode.REFUND_NOT_FOUND)
        if refund.status != 1:
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)

        order = await self.order_repository.get_by_id(db, refund.order_id)
        if not order:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        pay_method = order.pay_method or "balance"
        channel_refund_no = None
        refund_success = True
        error_message = None

        now = datetime.now()
        refund.audit_time = now
        refund.auditor_id = auditor_id
        refund.audit_remark = form.get("remark", "")
        refund.channel = pay_method

        try:
            if pay_method == "balance":
                await self.balance_account_service.refund(
                    db, order.user_id, refund.refund_amount
                )
            elif pay_method == "combined":
                balance_amount = order.balance_amount
                if balance_amount > 0:
                    balance_refund = refund.refund_amount * balance_amount // order.paid_amount
                    await self.balance_account_service.refund(db, order.user_id, balance_refund)
                    third_party_refund = refund.refund_amount - balance_refund
                    if third_party_refund > 0:
                        channel_refund_no = await self._channel_refund(
                            db, order, third_party_refund
                        )
                else:
                    channel_refund_no = await self._channel_refund(
                        db, order, refund.refund_amount
                    )
            else:
                channel_refund_no = await self._channel_refund(
                    db, order, refund.refund_amount
                )
        except Exception as e:
            logger.error("退款执行失败 refundId=%s: %s", refund_id, e)
            refund_success = False
            error_message = str(e)

        if refund_success:
            refund.status = 2
            refund.refund_time = now
            refund.channel_refund_no = channel_refund_no
            order.status = 6
            await self._rollback_fulfillment(db, order, refund)
        else:
            refund.status = 3
            refund.error_message = error_message
            order.status = _restore_status_for_order(order)

        await db.flush()
        await _invalidate_order_detail_cache(order.order_no)

        self.mongo_audit_log_repository.create_audit_async(
            operator_id=auditor_id,
            target_type="order",
            target_id=refund_id,
            action="refund_approve",
            module="order",
            after_value=form if not hasattr(form, "dict") else form,
        )

    async def _channel_refund(self, db: AsyncSession, order, refund_amount: int) -> str | None:
        # 组合支付需按支付流水中的实际第三方渠道退款（pay_method 为 combined 无单一渠道）
        payments = await self.payment_record_repository.list_by_order_id(db, order.id)
        if not payments:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "未找到支付流水，无法发起渠道退款")
        payment = payments[0]
        result = await self.payment_channel_service.refund(
            payment.channel,
            order.order_no,
            payment.payment_no,
            refund_amount,
            order.paid_amount,
        )
        if not result.success:
            raise BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, result.error_message or "渠道退款失败")
        return result.channel_refund_no

    async def _rollback_fulfillment(self, db: AsyncSession, order, refund) -> None:
        """退款成功后按商品类型回退履约。"""
        if order.package_type == "vip":
            await self.member_service.on_order_refunded(db, order, refund)
        elif order.package_type == "credit":
            used_credits = int(getattr(refund, "used_credits", 0) or 0)
            credit_amount = int(order.credit_amount or 0)
            unused = max(0, credit_amount - used_credits)
            if unused > 0:
                await self.ai_balance_service.deduct(db, order.user_id, unused)

    async def reject_refund(
        self, db: AsyncSession, refund_id: int, form: dict, auditor_id: int
    ) -> None:
        refund = await self.refund_record_repository.get_by_id(db, refund_id)
        if not refund:
            raise BusinessException(ResultCode.REFUND_NOT_FOUND)
        if refund.status != 1:
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)

        order = await self.order_repository.get_by_id(db, refund.order_id)
        if not order:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        refund.status = 3
        refund.audit_time = datetime.now()
        refund.auditor_id = auditor_id
        refund.audit_remark = form.get("remark", "")
        order.status = _restore_status_for_order(order)
        await db.flush()
        await _invalidate_order_detail_cache(order.order_no)

        self.mongo_audit_log_repository.create_audit_async(
            operator_id=auditor_id,
            target_type="order",
            target_id=refund_id,
            action="refund_reject",
            module="order",
            after_value=form if not hasattr(form, "dict") else form,
        )

    async def list_refunds(self, db: AsyncSession, query: dict) -> dict:
        items, total = await self.refund_record_repository.get_page(
            db,
            query["pageNum"],
            query["pageSize"],
            order_no=query.get("orderNo"),
            keywords=query.get("keywords"),
            status=query.get("status"),
            apply_time_start=query.get("applyTimeStart"),
            apply_time_end=query.get("applyTimeEnd"),
            reason_type=query.get("reasonType"),
        )
        list_data = [
            _refund_to_vo(item["refund"], item["order_no"], item.get("username") or "")
            for item in items
        ]
        return {"list": list_data, "total": total}

    async def retry_failed_refunds(self, db: AsyncSession) -> int:
        stmt = select(SysRefundRecord).where(
            SysRefundRecord.deleted == 0,
            SysRefundRecord.status == 3,
            SysRefundRecord.retry_count < REFUND_MAX_RETRY_COUNT,
        )
        result = await db.execute(stmt)
        failed_refunds = result.scalars().all()

        if not failed_refunds:
            return 0

        success_count = 0
        final_fail_count = 0
        for refund in failed_refunds:
            order = await self.order_repository.get_by_id(db, refund.order_id)
            if not order:
                logger.warning("退款重试跳过: 退款记录%s对应订单不存在", refund.id)
                continue

            refund.retry_count = (refund.retry_count or 0) + 1
            refund_success = True
            error_message = None

            try:
                if order.pay_method == "balance":
                    await self.balance_account_service.refund(
                        db, order.user_id, refund.refund_amount
                    )
                else:
                    channel_refund_no = await self._channel_refund(
                        db, order, refund.refund_amount
                    )
                    refund.channel_refund_no = channel_refund_no
            except Exception as e:
                logger.error("退款重试失败 refundId=%s: %s", refund.id, e)
                refund_success = False
                error_message = str(e)

            if refund_success:
                refund.status = 2
                refund.refund_time = datetime.now()
                refund.error_message = None
                order.status = 6
                await self._rollback_fulfillment(db, order, refund)
                success_count += 1
            else:
                if not error_message:
                    error_message = "渠道退款失败"
                if refund.retry_count >= REFUND_MAX_RETRY_COUNT:
                    error_message = f"{error_message}（已达重试上限，转为最终失败）"
                    final_fail_count += 1
                refund.error_message = error_message

            await db.flush()

        logger.debug(
            "退款失败重试完成: 总数=%s 成功=%s 最终失败=%s",
            len(failed_refunds),
            success_count,
            final_fail_count,
        )
        return len(failed_refunds)


refund_service = RefundService()
