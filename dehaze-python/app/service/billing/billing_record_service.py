"""计费记录与余额流水查询服务"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.ai_billing import (
    BillingRecordQuery,
    BillingRecordResult,
    CreditLogQuery,
    CreditLogResult,
)
from app.models.schema.common import PageResult
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_credit_log_repository import ai_credit_log_repository
from app.repository.ai_refund_repository import ai_refund_repository


class BillingRecordService:
    """计费记录 CRUD 与查询"""

    def __init__(
        self,
        ai_billing_repository=ai_billing_repository,
        ai_credit_log_repository=ai_credit_log_repository,
        ai_refund_repository=ai_refund_repository,
    ):
        self.ai_billing_repository = ai_billing_repository
        self.ai_credit_log_repository = ai_credit_log_repository
        self.ai_refund_repository = ai_refund_repository

    async def list_by_user(self, 
        db: AsyncSession,
        user_id: int,
        query: BillingRecordQuery,
    ) -> PageResult[BillingRecordResult]:
        records, total = await self.ai_billing_repository.list_by_user(
            db,
            user_id,
            query.page,
            query.size,
            conversation_id=query.conversation_id,
            bill_type=query.bill_type,
            model_id=query.model_id,
            date_start=query.date_start,
            date_end=query.date_end,
        )
        status_map = await self.ai_refund_repository.latest_status_by_billing_ids(
            db, [r.id for r in records]
        )
        results = []
        for r in records:
            result = BillingRecordResult.model_validate(r)
            result.refund_status = status_map.get(r.id, 0)
            results.append(result)
        return PageResult(
            list=results,
            total=total,
        )

    async def get_by_id(self, 
        db: AsyncSession,
        user_id: int,
        billing_id: int,
    ) -> BillingRecordResult:
        """查询单条计费记录（仅本人数据），不存在则抛资源不存在"""
        record = await self.ai_billing_repository.get_by_id(db, billing_id)
        if not record or record.user_id != user_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "计费记录不存在")
        return BillingRecordResult.model_validate(record)

    async def list_credit_logs(self, 
        db: AsyncSession,
        user_id: int,
        query: CreditLogQuery,
    ) -> PageResult[CreditLogResult]:
        logs, total = await self.ai_credit_log_repository.list_by_user(
            db,
            user_id,
            query.page,
            query.size,
            source=query.source,
            start=query.date_start,
            end=query.date_end,
        )
        return PageResult(
            list=[CreditLogResult.model_validate(log) for log in logs],
            total=total,
        )


billing_record_service = BillingRecordService()
