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


class BillingRecordService:
    """计费记录 CRUD 与查询"""

    @staticmethod
    async def list_by_user(
        db: AsyncSession,
        user_id: int,
        query: BillingRecordQuery,
    ) -> PageResult[BillingRecordResult]:
        records, total = await ai_billing_repository.list_by_user(
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
        return PageResult(
            list=[BillingRecordResult.model_validate(r) for r in records],
            total=total,
        )

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        user_id: int,
        billing_id: int,
    ) -> BillingRecordResult:
        """查询单条计费记录（仅本人数据），不存在则抛资源不存在"""
        record = await ai_billing_repository.get_by_id(db, billing_id)
        if not record or record.user_id != user_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "计费记录不存在")
        return BillingRecordResult.model_validate(record)

    @staticmethod
    async def list_credit_logs(
        db: AsyncSession,
        user_id: int,
        query: CreditLogQuery,
    ) -> PageResult[CreditLogResult]:
        logs, total = await ai_credit_log_repository.list_by_user(
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
