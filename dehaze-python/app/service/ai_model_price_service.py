"""模型用户售价管理服务：价格版本化 CRUD 与三维档位匹配换算积分"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_model_price import SysAiModelPrice
from app.models.schema.ai_model_price import (
    ModelPriceCreateRequest,
    ModelPriceDetailResult,
    ModelPriceQuery,
    ModelPriceResult,
)
from app.models.schema.common import PageResult
from app.core.timezone import is_peak_hour
from app.repository.ai_model_price_repository import ai_model_price_repository

logger = logging.getLogger(__name__)

_TZ = ZoneInfo("Asia/Shanghai")


class AiModelPriceService:
    """用户售价维护（价格版本化）与三维档位匹配积分换算"""

    def __init__(self, price_repository=ai_model_price_repository):
        self.price_repository = price_repository

    async def create_price(self, db: AsyncSession, request: ModelPriceCreateRequest) -> ModelPriceResult:
        """新增用户售价：同模型同供应商生成新价格版本，历史版本保留可追溯"""
        version = await self.price_repository.next_price_version(db, request.model_id, request.provider_id)
        price = await self.price_repository.create(
            db,
            SysAiModelPrice(
                model_id=request.model_id,
                provider_id=request.provider_id,
                price_version=version,
                unit=request.unit,
                effective_from=request.effective_from or datetime.now(_TZ),
                effective_to=request.effective_to,
                status=request.status,
            ),
        )
        details = await self.price_repository.create_details(
            db, price.id, [d.model_dump() for d in request.details]
        )
        return self._to_result(price, details)

    async def update_price(self, db: AsyncSession, price_id: int, data: dict) -> ModelPriceResult:
        """更新用户售价版本主表字段（单价单位/生效时间/状态）"""
        price = await self.price_repository.get_by_id(db, price_id)
        if price is None:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户售价不存在")
        if data:
            await self.price_repository.update(db, price, data)
        details = await self.price_repository.list_details(db, price.id)
        return self._to_result(price, details)

    async def delete_price(self, db: AsyncSession, price_id: int) -> None:
        """删除用户售价版本（主表与档位明细逻辑删除）"""
        price = await self.price_repository.get_by_id(db, price_id)
        if price is None:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户售价不存在")
        await self.price_repository.soft_delete_by_ids(db, [price_id])
        await self.price_repository.soft_delete_details_by_price_id(db, price_id)

    async def list_prices(self, db: AsyncSession, query: ModelPriceQuery) -> PageResult[ModelPriceResult]:
        prices, total = await self.price_repository.list_prices(
            db,
            query.page,
            query.size,
            model_id=query.model_id,
            provider_id=query.provider_id,
        )
        results = []
        for price in prices:
            details = await self.price_repository.list_details(db, price.id)
            results.append(self._to_result(price, details))
        return PageResult(list=results, total=total)

    async def calculate(
        self,
        db: AsyncSession,
        model_id: str,
        provider_id: int,
        at_time: datetime,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
    ) -> dict:
        """按调用时刻换算用户积分：价格版本 → 时段档位 → 上下文分段 三维匹配（积分）

        未配置用户售价返回 {"credits": 0, "credits_saved": 0}（由调用方标记待配置）；
        换算公式见后端实现.md §2.12：
          credits = (input - cached) × input单价/1M + cached × cached单价/1M + output × output单价/1M
          credits_saved = cached × (input单价 - cached单价) / 1M
        Decimal 精确计算后四舍五入取整，单次至少 1 积分。
        """
        version = await self.price_repository.get_effective_version(
            db, model_id, provider_id, at_time
        )
        if version is None:
            # 未配置价格版本：configured=False 供调用方区分"未配置"与"免费（全0价）"
            return {"credits": 0, "credits_saved": 0, "configured": False}
        details = await self.price_repository.list_details(db, version.id)
        slot = "peak" if is_peak_hour(at_time) else "idle"
        total_input = input_tokens + cached_tokens

        def _unit_price(token_type: str) -> Decimal:
            for d in details:
                if (
                    d.token_type == token_type
                    and d.time_slot == slot
                    and d.min_tokens <= total_input
                    and (d.max_tokens is None or total_input < d.max_tokens)
                ):
                    return d.unit_price
            return Decimal("0")

        uncached = max(input_tokens - cached_tokens, 0)
        input_price = _unit_price("input")
        cached_price = _unit_price("cached")
        output_price = _unit_price("output")
        if input_price == 0 and cached_price == 0 and output_price == 0:
            # 全 0 价配置（如内置本地免费模型）明确不扣积分
            return {"credits": 0, "credits_saved": 0, "configured": True}
        credits = (
            uncached * input_price
            + cached_tokens * cached_price
            + output_tokens * output_price
        ) / Decimal("1000000")
        credits = credits.to_integral_value(rounding=ROUND_HALF_UP)
        credits_saved = (
            cached_tokens * (input_price - cached_price) / Decimal("1000000")
        ).to_integral_value(rounding=ROUND_HALF_UP)
        return {
            "credits": max(int(credits), 1),
            "credits_saved": max(int(credits_saved), 0),
            "configured": True,
        }

    @staticmethod
    def _to_result(price, details) -> ModelPriceResult:
        result = ModelPriceResult(
            id=price.id,
            model_id=price.model_id,
            provider_id=price.provider_id,
            price_version=price.price_version,
            unit=price.unit,
            effective_from=price.effective_from,
            effective_to=price.effective_to,
            status=price.status,
            create_time=price.create_time,
            update_time=price.update_time,
        )
        result.details = [ModelPriceDetailResult.model_validate(d) for d in details]
        return result


ai_model_price_service = AiModelPriceService()
