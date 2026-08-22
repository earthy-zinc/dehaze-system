"""计费统计服务（管理员）"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema.ai_billing import BillingStatQuery, BillingStatResult
from app.repository.ai_billing_repository import ai_billing_repository


class BillingStatService:
    """按 user/model/billType/day 维度统计"""

    async def stats(self, 
        db: AsyncSession, query: BillingStatQuery, user_id: int | None = None
    ) -> list[BillingStatResult]:
        rows = await ai_billing_repository.stats_by_dimension(
            db,
            query.group_by,
            user_id=user_id,
            model_id=query.model_id,
            bill_type=query.bill_type,
            start=query.date_start,
            end=query.date_end,
        )
        results = []
        for row in rows:
            total_input = row["total_input_tokens"]
            cache_hit_rate = (
                round(row["cached_input_tokens"] / total_input, 4) if total_input > 0 else 0.0
            )
            results.append(
                BillingStatResult(
                    dimension=row["dimension"],
                    total_credits=row["total_credits"],
                    total_input_tokens=row["total_input_tokens"],
                    total_output_tokens=row["total_output_tokens"],
                    cache_hit_rate=cache_hit_rate,
                    credits_saved=row["credits_saved"],
                    degradation_count=row["degradation_count"],
                )
            )
        return results


billing_stat_service = BillingStatService()
