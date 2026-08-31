"""余额充值域：充值下单与回调入账（需求规格 §3.5.3）。

充值面向人民币余额账户（微信/支付宝），与积分卡购买（sys_order + AI 计费到账）隔离：
渠道统一下单以充值单号为商户单号，回调成功后可用余额入账并写 recharge 流水。
"""

import logging
import random
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_recharge import SysRecharge
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.repository.recharge_repository import recharge_repository
from app.service.payment_channel_service import payment_channel_service

logger = logging.getLogger(__name__)

PAYMENT_LOCK_TTL = 10


def _gen_recharge_no() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    rand = random.randint(100000, 999999)
    return f"RC{ts}{rand}"


class RechargeService:
    def __init__(
        self,
        recharge_repository=recharge_repository,
        payment_channel_service=payment_channel_service,
        mongo_audit_log_repository=mongo_audit_log_repository,
        balance_account_service=None,
    ):
        self.recharge_repository = recharge_repository
        self.payment_channel_service = payment_channel_service
        self.mongo_audit_log_repository = mongo_audit_log_repository
        if balance_account_service is None:
            from app.service.order.balance_account_service import (
                balance_account_service as _b,
            )

            balance_account_service = _b
        self.balance_account_service = balance_account_service

    async def create_recharge(self, db: AsyncSession, form: dict, user_id: int) -> dict:
        pay_method = form["payMethod"]
        if pay_method not in ("wechat", "alipay"):
            raise BusinessException(ResultCode.PARAM_ERROR, "充值仅支持微信/支付宝支付")

        recharge_no = _gen_recharge_no()
        pay_result = await self.payment_channel_service.unified_order(
            pay_method, recharge_no, form["amount"], "余额充值"
        )
        record = SysRecharge(
            recharge_no=recharge_no,
            user_id=user_id,
            amount=form["amount"],
            pay_method=pay_method,
            status=1,
        )
        await self.recharge_repository.create(db, record)

        self.mongo_audit_log_repository.create_audit_async(
            operator_id=user_id,
            target_type="recharge",
            target_id=recharge_no,
            action="create",
            module="order",
            after_value={"amount": form["amount"], "payMethod": pay_method},
        )
        return {
            "rechargeNo": recharge_no,
            "payMethod": pay_method,
            "payUrl": pay_result.pay_url,
            "qrCode": pay_result.qr_code,
        }

    async def handle_payment_callback(self, db: AsyncSession, callback, channel: str) -> bool:
        """充值回调：金额校验 + 幂等 + 余额入账。

        订单回调链路未命中 sys_order 时路由到此；非充值单号返回 False。
        """
        from app.infrastructure.cache.redis_lock import acquire_lock, release_lock

        recharge = await self.recharge_repository.get_by_recharge_no(db, callback.order_no)
        if not recharge:
            return False

        if callback.amount != recharge.amount:
            logger.error(
                "充值回调金额不一致 rechargeNo=%s expected=%s actual=%s",
                recharge.recharge_no,
                recharge.amount,
                callback.amount,
            )
            raise BusinessException(ResultCode.PAYMENT_AMOUNT_MISMATCH)

        lock_key = f"payment:lock:{recharge.recharge_no}"
        lock_token = await acquire_lock(lock_key, PAYMENT_LOCK_TTL)
        if lock_token is None:
            logger.info("充值回调获取锁失败，幂等返回 rechargeNo=%s", recharge.recharge_no)
            return True

        try:
            if recharge.status == 2:
                logger.info("充值回调幂等返回 rechargeNo=%s", recharge.recharge_no)
                return True
            if recharge.status != 1:
                logger.warning(
                    "充值回调状态异常 rechargeNo=%s status=%s", recharge.recharge_no, recharge.status
                )
                return False

            recharge.status = 2
            recharge.channel_payment_no = callback.channel_payment_no
            recharge.pay_time = datetime.now()
            await db.flush()
            await self.balance_account_service.recharge(db, recharge.user_id, recharge.amount)
        finally:
            await release_lock(lock_key, lock_token)

        self.mongo_audit_log_repository.create_audit_async(
            operator_id=recharge.user_id,
            target_type="recharge",
            target_id=recharge.recharge_no,
            action="paid",
            module="order",
            after_value={"amount": recharge.amount, "channel": channel},
        )
        return True


recharge_service = RechargeService()
