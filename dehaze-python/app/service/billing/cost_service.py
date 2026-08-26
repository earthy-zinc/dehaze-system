"""成本管理服务：成本单价维护（价格版本化）、成本核算回填、供应商账单对账"""

import logging
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.timezone import is_peak_hour
from app.models.entity.sys_ai_model_cost import SysAiModelCost
from app.models.schema.ai_billing_cost import (
    ModelCostCreateRequest,
    ModelCostDetailResult,
    ModelCostQuery,
    ModelCostResult,
)
from app.models.schema.common import PageResult
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_model_cost_repository import ai_model_cost_repository

logger = logging.getLogger(__name__)

_TZ = ZoneInfo("Asia/Shanghai")


class CostService:
    """成本单价维护（价格版本化）与成本核算（三维档位匹配）"""

    def __init__(
        self,
        ai_model_cost_repository=ai_model_cost_repository,
        ai_billing_repository=ai_billing_repository,
    ):
        self.cost_repository = ai_model_cost_repository
        self.billing_repository = ai_billing_repository

    async def create_cost(self, db: AsyncSession, request: ModelCostCreateRequest) -> ModelCostResult:
        """新增成本单价：同模型同供应商生成新的价格版本，历史版本保留可追溯"""
        version = await self.cost_repository.next_price_version(db, request.model_id, request.provider_id)
        cost = await self.cost_repository.create(
            db,
            SysAiModelCost(
                model_id=request.model_id,
                provider_id=request.provider_id,
                price_version=version,
                currency=request.currency,
                effective_from=request.effective_from or datetime.now(_TZ),
                effective_to=request.effective_to,
                status=request.status,
            ),
        )
        details = await self.cost_repository.create_details(
            db, cost.id, [d.model_dump() for d in request.details]
        )
        return self._to_result(cost, details)

    async def update_cost(self, db: AsyncSession, cost_id: int, data: dict) -> ModelCostResult:
        """更新成本价格版本主表字段（币种/生效时间/状态）"""
        cost = await self.cost_repository.get_by_id(db, cost_id)
        if cost is None:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "成本单价不存在")
        if data:
            await self.cost_repository.update(db, cost, data)
        details = await self.cost_repository.list_details(db, cost.id)
        return self._to_result(cost, details)

    async def delete_cost(self, db: AsyncSession, cost_id: int) -> None:
        """删除成本单价（主表与档位明细逻辑删除）"""
        cost = await self.cost_repository.get_by_id(db, cost_id)
        if cost is None:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "成本单价不存在")
        await self.cost_repository.soft_delete_by_ids(db, [cost_id])
        await self.cost_repository.soft_delete_details_by_price_id(db, cost_id)

    async def list_costs(self, db: AsyncSession, query: ModelCostQuery) -> PageResult[ModelCostResult]:
        costs, total = await self.cost_repository.list_costs(
            db,
            query.page,
            query.size,
            keyword=query.keyword,
            model_id=query.model_id,
            provider_id=query.provider_id,
        )
        results = []
        for cost in costs:
            details = await self.cost_repository.list_details(db, cost.id)
            results.append(self._to_result(cost, details))
        return PageResult(list=results, total=total)

    async def calculate_cost(
        self,
        db: AsyncSession,
        model_id: str,
        provider_id: int,
        at_time: datetime,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        """按调用时刻核算成本：价格版本 → 时段档位 → 上下文分段 三维匹配（元）

        未配置成本单价返回 0（由调用方标记待配置）。
        """
        version = await self.cost_repository.get_effective_version(
            db, model_id, provider_id, at_time
        )
        if version is None:
            return Decimal("0")
        details = await self.cost_repository.list_details(db, version.id)
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
        cost = (
            uncached * _unit_price("input")
            + cached_tokens * _unit_price("cached")
            + output_tokens * _unit_price("output")
        ) / Decimal("1000000")
        return cost.quantize(Decimal("0.0001"))

    async def backfill_cost(self, db: AsyncSession, billing_id: int) -> None:
        """按计费记录回填成本（cost 线异步核算入口，未配置成本价置 0）"""
        billing = await self.billing_repository.get_by_id(db, billing_id)
        if billing is None or billing.provider_id is None:
            return
        cost = await self.calculate_cost(
            db,
            billing.model,
            billing.provider_id,
            billing.create_time,
            billing.input_tokens,
            billing.cached_input_tokens,
            billing.output_tokens,
        )
        await self.billing_repository.update(db, billing, {"cost": cost})

    @staticmethod
    def import_reconcile(content: str) -> int:
        """供应商账单导入（最小实现：按非空行计数，对账骨架）"""
        return len([line for line in content.splitlines() if line.strip()])

    @staticmethod
    def _to_result(cost, details) -> ModelCostResult:
        result = ModelCostResult(
            id=cost.id,
            model_id=cost.model_id,
            provider_id=cost.provider_id,
            price_version=cost.price_version,
            currency=cost.currency,
            effective_from=cost.effective_from,
            effective_to=cost.effective_to,
            status=cost.status,
            create_time=cost.create_time,
            update_time=cost.update_time,
        )
        result.details = [ModelCostDetailResult.model_validate(d) for d in details]
        return result


cost_service = CostService()
