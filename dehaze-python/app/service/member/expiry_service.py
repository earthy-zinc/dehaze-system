"""到期处理域：会员过期降级与到期预警提醒。"""

import logging
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_member import SysMember
from app.repository.member_benefit_repository import member_benefit_repository
from app.service.member.member_service import (
    _apply_benefit_quotas,
    _calculate_level,
    _invalidate_member_cache,
)

logger = logging.getLogger(__name__)


class MemberExpiryService:
    def __init__(self, member_benefit_repository=member_benefit_repository):
        self.member_benefit_repository = member_benefit_repository

    async def process_expired_members(self, db: AsyncSession) -> int:
        """会员过期降级处理

        扫描 expire_time < NOW() AND level_source != 'growth' 的会员，
        按成长值重算等级、置 level_source='growth'、清空 expire_time、刷新权益。

        Returns:
            已处理的会员数量
        """
        now = datetime.now()
        stmt = select(SysMember).where(
            SysMember.deleted == 0,
            SysMember.expire_time.isnot(None),
            SysMember.expire_time < now,
            SysMember.level_source != "growth",
        )
        result = await db.execute(stmt)
        members = result.scalars().all()

        if not members:
            return 0

        benefits = await self.member_benefit_repository.list_ordered_by_growth_min(db)
        benefit_map = {b.level_code: b for b in benefits}

        count = 0
        for member in members:
            old_level = member.level_code
            target_level = _calculate_level(benefits, member.growth_value)
            member.level_code = target_level
            member.level_source = "growth"
            member.expire_time = None
            benefit = benefit_map.get(target_level)
            if benefit:
                _apply_benefit_quotas(member, benefit)
            count += 1
            await _invalidate_member_cache(user_id=member.user_id, level_code=old_level)
            await _invalidate_member_cache(level_code=target_level)

            if target_level != old_level:
                try:
                    from app.service.message_service import message_service

                    old_benefit = benefit_map.get(old_level)
                    new_benefit = benefit_map.get(target_level)
                    await message_service.send(
                        db,
                        {
                            "type": "member",
                            "recipientIds": [member.user_id],
                            "bizModule": "member",
                            "bizId": f"level_change:{member.user_id}:{int(now.timestamp())}",
                            "templateCode": "member_downgrade_warning",
                            "variables": {
                                "currentLevel": old_benefit.level_name
                                if old_benefit
                                else old_level,
                                "days": "0",
                                "downgradeLevel": new_benefit.level_name
                                if new_benefit
                                else target_level,
                            },
                        },
                    )
                except Exception as e:
                    logger.warning(
                        (
                            f"等级变更通知发送失败: userId={member.user_id}, "
                            f"old={old_level}, new={target_level}"
                        ),
                        exc_info=e,
                    )

        await db.flush()
        logger.debug(f"会员过期降级处理完成: 共处理 {count} 条记录")
        return count

    async def send_expire_reminders(self, db: AsyncSession) -> int:
        from app.service.message_service import message_service

        now = datetime.now()
        benefits = await self.member_benefit_repository.list_ordered_by_growth_min(db)
        benefit_map = {b.level_code: b for b in benefits}

        day_template_map = {
            7: ("expire_reminder_7d", "member_expire_reminder_7"),
            3: ("expire_reminder_3d", "member_expire_reminder_3"),
            1: ("expire_reminder_1d", "member_expire_reminder_1"),
        }

        sent_count = 0
        for days, (biz_prefix, template_code) in day_template_map.items():
            window_start = (now + timedelta(days=days)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            window_end = window_start + timedelta(days=1)

            stmt = select(SysMember).where(
                SysMember.deleted == 0,
                SysMember.expire_time.isnot(None),
                SysMember.expire_time >= window_start,
                SysMember.expire_time < window_end,
                SysMember.level_source != "growth",
            )
            result = await db.execute(stmt)
            members = result.scalars().all()
            if not members:
                continue

            for member in members:
                try:
                    current_benefit = benefit_map.get(member.level_code)
                    variables = {
                        "currentLevel": current_benefit.level_name
                        if current_benefit
                        else member.level_code,
                        "days": str(days),
                        "expireDate": member.expire_time.strftime("%Y-%m-%d")
                        if member.expire_time
                        else "",
                    }
                    if days == 3:
                        target_level = _calculate_level(benefits, member.growth_value)
                        downgrade_benefit = benefit_map.get(target_level)
                        variables["downgradeLevel"] = (
                            downgrade_benefit.level_name if downgrade_benefit else target_level
                        )
                        if current_benefit and downgrade_benefit:
                            variables["benefitCompare"] = (
                                f"去雾:{current_benefit.monthly_dehaze_quota}→{downgrade_benefit.monthly_dehaze_quota}次/月，"
                                f"评估:{current_benefit.monthly_evaluate_quota}→{downgrade_benefit.monthly_evaluate_quota}次/月"
                            )
                        else:
                            variables["benefitCompare"] = ""

                    await message_service.send(
                        db,
                        {
                            "type": "member",
                            "recipientIds": [member.user_id],
                            "bizModule": "member",
                            "bizId": f"{biz_prefix}:{member.user_id}:{now.strftime('%Y-%m-%d')}",
                            "templateCode": template_code,
                            "variables": variables,
                        },
                    )
                    sent_count += 1
                except Exception as e:
                    logger.warning(
                        f"到期提醒发送失败: userId={member.user_id}, days={days}", exc_info=e
                    )

        logger.debug(f"会员到期预警完成: 共发送 {sent_count} 条提醒")
        return sent_count


member_expiry_service = MemberExpiryService()
