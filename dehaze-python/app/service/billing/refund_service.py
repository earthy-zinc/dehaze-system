"""退款补偿服务"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.ai_billing import RefundResult
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_refund_repository import ai_refund_repository
from app.service.billing.balance_service import balance_service


class RefundService:
    """退款补偿：申请 → 审核 → 余额回补"""

    async def apply_refund(self, 
        db: AsyncSession,
        user_id: int,
        billing_id: int,
        amount: int,
        reason: str,
    ) -> RefundResult:
        """用户申请退款
        1. 校验计费记录存在且属于用户
        2. 校验未重复申请
        3. 创建退款申请记录（状态：待审核）
        """
        if amount <= 0:
            raise BusinessException(ResultCode.PARAM_ERROR, "退款积分数必须大于 0")
        record = await ai_billing_repository.get_by_id(db, billing_id)
        if not record or record.user_id != user_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "计费记录不存在")
        existing = await ai_refund_repository.get_pending_by_billing_id(db, billing_id)
        if existing:
            raise BusinessException(ResultCode.AI_REFUND_ALREADY_EXISTS)
        refund = await ai_refund_repository.create_refund(
            db,
            user_id=user_id,
            billing_id=billing_id,
            amount=amount,
            reason=reason,
            status=1,  # 待审核
            create_by=user_id,
        )
        return RefundResult.model_validate(refund)

    async def audit_refund(self, 
        db: AsyncSession,
        refund_id: int,
        approved: bool,
        audit_reason: str | None,
        operator_id: int,
    ) -> RefundResult:
        """管理员审核退款
        approved=True: 余额回补（source=refund, related_id=billing_id），不重置日/月限额已用计数
        approved=False: 标记拒绝
        """
        refund = await ai_refund_repository.get_by_id(db, refund_id)
        if not refund:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "退款申请不存在")
        if refund.status != 1:  # 非待审核
            raise BusinessException(ResultCode.REFUND_AUDIT_FAILED, "该退款申请已审核")

        if approved:
            # 余额回补（Redis INCR + MySQL CAS + 流水），不调整配额已用计数
            await balance_service.increase(
                db,
                refund.user_id,
                refund.amount,
                source="refund",
                related_id=refund.billing_id,
                reason=f"退款: {refund.reason}",
                operator_id=operator_id,
            )
            refund.status = 2  # 已通过
        else:
            refund.status = 3  # 已驳回
        refund.auditor_id = operator_id
        refund.audit_remark = audit_reason
        await db.flush()
        await db.refresh(refund)
        return RefundResult.model_validate(refund)


refund_service = RefundService()
