"""AI 知识库计费服务：embedding/rerank 按输入 token 后扣费

与 AI 对话计费（billing_service 预扣-结算两阶段）不同，KB 场景单次金额小且
发生在检索/文档处理动作内部，采用语音同款"调用前预估校验 + 调用后实扣"：
- ensure：调用前校验欠费/配额/余额（fail-closed），不满足抛 QUOTA_INSUFFICIENT
  拒绝（检索链路对该错误码透传而非降级，见 search_service._retrieve_with_timeout）
- charge_embedding / charge_rerank：模型调用成功后按实际 token 实扣并落
  sys_ai_billing；调用失败不扣（文档向量化失败由上游标记文档 failed）

口径（AI计费管理 后端实现 §KB 计费）：
- bill_type=embedding/rerank，token 数记入 input_tokens 作用量字段（与 asr/tts
  记秒/字符同款复用）；统计不进 CHAT_BILL_TYPES（chat token 列），月结账单与
  credits 明细按 bill_type 全量覆盖
- 单价读 sys_ai_model_price 用户售价（与 chat 同源同换算函数）；未配置售价
  warn 回退 0（免费）不拒绝——文档向量化是后台任务，计费配置缺失不应让文档
  永久卡在失败
- provider_code=local（内置本地模型零边际成本）不产生计费记录，但预校验
  照常执行（欠费/权益缺失与 chat 同口径拦截）
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_credit_log_repository import ai_credit_log_repository
from app.repository.ai_provider_repository import ai_provider_repository
from app.service.ai_model_price_service import ai_model_price_service
from app.service.billing.balance_service import balance_service
from app.service.billing.quota_service import quota_service

logger = logging.getLogger(__name__)

_LOCAL_PROVIDER_CODE = "local"

# bill_type → 消费流水 reason 文案
_BILL_REASONS = {"embedding": "知识库向量化", "rerank": "知识库重排序"}


class KbBillingService:
    """知识库向量化/重排序计费：预估校验 + 实扣落账"""

    def __init__(
        self,
        ai_billing_repository=ai_billing_repository,
        ai_credit_log_repository=ai_credit_log_repository,
        ai_provider_repository=ai_provider_repository,
        ai_model_price_service=ai_model_price_service,
        balance_service=balance_service,
        quota_service=quota_service,
    ):
        self.ai_billing_repository = ai_billing_repository
        self.ai_credit_log_repository = ai_credit_log_repository
        self.ai_provider_repository = ai_provider_repository
        self.ai_model_price_service = ai_model_price_service
        self.balance_service = balance_service
        self.quota_service = quota_service

    async def ensure(self, user_id: int, provider_code: str, model: str, tokens: int) -> None:
        """调用前预估校验：欠费熔断/配额/余额任一不满足抛 QUOTA_INSUFFICIENT"""
        async with get_db_session() as db:
            estimated = await self._calc_credits(db, provider_code, model, tokens)
            if await self.balance_service.is_arrears(user_id):
                raise BusinessException(ResultCode.QUOTA_INSUFFICIENT, "账户欠费，请充值后继续使用")
            if not await self.quota_service.check_quota(db, user_id, estimated):
                raise BusinessException(
                    ResultCode.QUOTA_INSUFFICIENT,
                    "今日或本月 AI 积分配额不足，请升级会员或明日再试",
                )
            if not await self.balance_service.check_balance(db, user_id, estimated):
                raise BusinessException(
                    ResultCode.QUOTA_INSUFFICIENT, "积分余额不足，请充值后继续使用"
                )

    async def charge_embedding(
        self, user_id: int, provider_code: str, model: str, tokens: int
    ) -> int:
        """文档/查询向量化实扣（bill_type=embedding），返回消耗积分"""
        return await self.charge(user_id, provider_code, model, tokens, bill_type="embedding")

    async def charge_rerank(self, user_id: int, provider_code: str, model: str, tokens: int) -> int:
        """检索重排序实扣（bill_type=rerank），返回消耗积分"""
        return await self.charge(user_id, provider_code, model, tokens, bill_type="rerank")

    async def charge(
        self,
        user_id: int,
        provider_code: str,
        model: str,
        tokens: int,
        *,
        bill_type: str,
    ) -> int:
        """模型调用成功后实扣配额与余额并落计费记录（voice 后扣费同款）

        local 模型完全跳过不落记录；0 积分（免费/未配置售价）仅落归因记录不扣减。
        """
        if provider_code == _LOCAL_PROVIDER_CODE:
            return 0
        async with get_db_session() as db:
            credits = await self._calc_credits(db, provider_code, model, tokens)
            provider = await self.ai_provider_repository.get_by_provider_code(db, provider_code)

            if credits > 0:
                await self.quota_service.deduct(user_id, credits)
                await self.balance_service.deduct(db, user_id, credits)

            billing = await self.ai_billing_repository.create_billing(
                db,
                user_id=user_id,
                model=model,
                bill_type=bill_type,
                input_tokens=tokens,
                credits=credits,
                quota_consumed=credits,
                pre_deduct=0,
                provider_id=provider.id if provider else None,
            )

            if credits > 0:
                balance = await self.balance_service.get_balance(db, user_id)
                await self.ai_credit_log_repository.create_log(
                    db,
                    user_id=user_id,
                    source="consume",
                    amount=Decimal(-credits),
                    balance_after=balance,
                    related_id=billing.id,
                    reason=f"{_BILL_REASONS[bill_type]}消耗（{model}）",
                )

            logger.info(
                "KB 计费完成: user_id=%s bill_type=%s model=%s tokens=%s credits=%s",
                user_id,
                bill_type,
                model,
                tokens,
                credits,
            )
            return credits

    async def _calc_credits(self, db: Any, provider_code: str, model: str, tokens: int) -> int:
        """按用户售价换算积分；未配置售价 warn 回退 0（免费，防误扣）"""
        if provider_code == _LOCAL_PROVIDER_CODE:
            return 0
        provider = await self.ai_provider_repository.get_by_provider_code(db, provider_code)
        calc = await self.ai_model_price_service.calculate(
            db, model, provider.id if provider else None, datetime.now(), tokens, 0, 0
        )
        if not calc["configured"]:
            logger.warning("模型 %s 未配置用户售价，KB 计费按 0 积分（免费）处理", model)
        return calc["credits"]


kb_billing_service = KbBillingService()
