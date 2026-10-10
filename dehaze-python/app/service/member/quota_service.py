"""会员权益配额域：解析 8 类任务的生效配额，并在会员卡履约/权益变更时刷新会员配额字段。

图像任务消费（校验/扣减/归还）与月度重置已随 D1 全积分制退役，改按积分计费；
配额字段与刷新逻辑保留至「会员配额字段从 API 面移除」独立任务落地，以维持三端口径一致。
"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_member import QUOTA_TASK_TYPES, SysMember


def _effective_task_quota(benefit, overrides: dict | None) -> dict[str, int]:
    """合并等级权益与会员卡覆盖项，得到各任务的生效配额（覆盖与权益取较高值）。"""
    result = {}
    for task_type in QUOTA_TASK_TYPES:
        base = getattr(benefit, f"monthly_{task_type}_quota", 0) or 0
        if overrides:
            # overrides 的 key 为 camelCase（套餐侧写入/DB/Java 序列化统一口径，
            # 如 monthlySuperResolutionQuota）
            camel = "".join(part.capitalize() for part in task_type.split("_"))
            base = max(base, int(overrides.get(f"monthly{camel}Quota", 0) or 0))
        result[task_type] = base
    return result


def resolve_card_overrides(member: SysMember, package) -> dict | None:
    """已购会员卡（level_source=purchase 且未到期）生效的覆盖项，其余回退等级权益。"""
    if member.level_source != "purchase":
        return None
    if member.expire_time is not None and member.expire_time < datetime.now():
        return None
    if package is None or not package.benefit_overrides:
        return None
    return package.benefit_overrides


class MemberQuotaService:
    async def refresh_member_quota(self, db: AsyncSession, member: SysMember, benefit) -> None:
        """会员卡履约/等级联动时刷新会员 8 类任务配额（不含已用量）"""
        for task_type in QUOTA_TASK_TYPES:
            setattr(
                member, f"monthly_{task_type}_quota", getattr(benefit, f"monthly_{task_type}_quota")
            )
        await db.flush()


member_quota_service = MemberQuotaService()
