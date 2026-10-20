"""计费统计服务（管理员统计 + 用户端消耗汇总）"""

from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.ai_billing import (
    BillingModelDistResult,
    BillingSavingsResult,
    BillingStatQuery,
    BillingStatResult,
    BillingSummaryResult,
    BillingTrendPointResult,
)
from app.repository.ai_billing_repository import CHAT_BILL_TYPES, ai_billing_repository


class BillingStatService:
    """按 user/model/billType/day 维度统计（管理员）；按日/月消耗汇总（用户端）"""

    def __init__(self, ai_billing_repository=ai_billing_repository):
        self.ai_billing_repository = ai_billing_repository

    async def stats(
        self, db: AsyncSession, query: BillingStatQuery, user_id: int | None = None
    ) -> list[BillingStatResult]:
        # 白名单须与 repository.stats_by_dimension 支持的维度保持一致；不校验的话
        # 非法值会在 repository 抛 ValueError，以 500 泄漏内部异常而非业务码
        if query.group_by not in ("user", "model", "billType", "day"):
            raise BusinessException(
                ResultCode.PARAM_ERROR, "groupBy 仅支持 user/model/billType/day"
            )
        rows = await self.ai_billing_repository.stats_by_dimension(
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
            chat_input = row["total_input_tokens"]
            cache_hit_rate = (
                round(row["chat_cached_tokens"] / chat_input, 4) if chat_input > 0 else 0.0
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

    async def summary(
        self,
        db: AsyncSession,
        user_id: int,
        dimension: str,
    ) -> BillingSummaryResult:
        """用户端消耗汇总：当前时段（日/月）总消耗、趋势、模型分布、节省汇总。

        仅统计 chat 类记录（主图 chat + 子 Agent chat_subagent，子 Agent 消耗
        实际扣减了用户配额与余额，须与扣减口径一致；asr/tts 的 input_tokens 存储
        秒数/字符数，语音/对话计量口径不同）；仅返回本人数据（收入线），不含任何成本字段。
        """
        now = datetime.now()
        # create_time 以秒级精度入库（MySQL DATETIME 对微秒做四舍五入），
        # 若上界保留微秒，刚写入的记录被进位到下一秒后会被 `<= now` 排除。
        # 上界取下一整秒，确保包含截止此刻的全部记录。
        now_ceiling = (now + timedelta(seconds=1)).replace(microsecond=0)
        if dimension == "month":
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif dimension == "day":
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            raise BusinessException(ResultCode.PARAM_ERROR, "dimension 仅支持 day/month")

        rows = await self.ai_billing_repository.sum_credits_by_user_group_by_period(
            db, user_id, period_start, now_ceiling, period=dimension, bill_type=CHAT_BILL_TYPES
        )
        trend = [
            BillingTrendPointResult(
                date=r["date"],
                credits=r["credits"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
            )
            for r in rows
        ]

        dist_rows = await self.ai_billing_repository.sum_credits_by_user_group_by_model(
            db, user_id, period_start, now_ceiling, bill_type=CHAT_BILL_TYPES
        )
        dist_rows.sort(key=lambda r: r["credits"], reverse=True)
        model_distribution = [
            BillingModelDistResult(
                model=r["model"],
                credits=r["credits"],
                tokens=r["input_tokens"] + r["output_tokens"],
            )
            for r in dist_rows[:5]
        ]

        return BillingSummaryResult(
            total_credits=sum(r["credits"] for r in rows),
            input_tokens=sum(r["input_tokens"] for r in rows),
            output_tokens=sum(r["output_tokens"] for r in rows),
            trend=trend,
            model_distribution=model_distribution,
            savings=BillingSavingsResult(
                cached_input_tokens=sum(r["cached_input_tokens"] for r in rows),
                credits_saved=sum(r["credits_saved"] for r in rows),
            ),
        )


billing_stat_service = BillingStatService()
