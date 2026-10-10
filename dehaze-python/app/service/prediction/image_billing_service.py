"""图像任务计费服务：按张计入 AI 积分（统一模型价格表，设计 §2.6 阶段 3）

图像任务调用前做余额预校验（不预扣），成功后按张实扣并记 `sys_ai_billing`
（`bill_type='image'`、`quantity_unit='image'`）与积分流水；缓存/WPX 命中与
失败/取消任务不计费（对齐 TTS 缓存命中不计费）。评估（PSNR/SSIM）为平台免费
能力，不经本服务。计费执行复用 unit_billing（与语音计费共用）。
"""

from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_credit_log_repository import ai_credit_log_repository
from app.service.ai_model_price_service import ai_model_price_service
from app.service.billing.balance_service import balance_service
from app.service.billing.quota_service import quota_service
from app.service.billing.unit_billing import UnitBillingService


class ImageBillingService(UnitBillingService):
    """图像生成/编辑能力计费：按成功张数计（经统一模型价格表）"""

    def __init__(
        self,
        ai_billing_repository=ai_billing_repository,
        ai_credit_log_repository=ai_credit_log_repository,
        balance_service=balance_service,
        quota_service=quota_service,
        price_service=None,
    ):
        super().__init__(
            ai_billing_repository=ai_billing_repository,
            ai_credit_log_repository=ai_credit_log_repository,
            balance_service=balance_service,
            quota_service=quota_service,
            price_service=price_service if price_service is not None else ai_model_price_service,
        )

    async def estimate_image_credits(
        self, db: AsyncSession, model_id: str, provider_id: int, image_count: int = 1
    ) -> int:
        """图像预估积分（按张，用于提交前预校验）"""
        return await self.estimate_credits(db, model_id, provider_id, Decimal(image_count))

    async def charge_image(
        self,
        db: AsyncSession,
        user_id: int,
        model_id: str,
        provider_id: int,
        image_count: int = 1,
    ) -> int:
        """按成功张数实扣，返回消耗积分数"""
        return await self._charge(
            db,
            user_id,
            bill_type="image",
            model_id=model_id,
            provider_id=provider_id,
            quantity=Decimal(image_count),
            quantity_unit="image",
            raw_quantity=image_count,
            reason=f"图像处理消耗（{model_id}）",
        )


image_billing_service = ImageBillingService()
