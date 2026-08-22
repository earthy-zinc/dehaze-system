"""
XXL-Job 定时任务 Handler

定义所有注册到 XXL-Job 调度中心的定时任务。
任务在 XXL-Job Admin 控制台中配置 CRON 表达式和参数。

任务清单：
- cleanupExpiredTasks:     清理过期任务（每天凌晨 2 点）
- cleanupStuckTasks:       回收僵死任务（每小时）
- modelHealthCheck:        模型健康检查（每 30 分钟）
- cleanupOrphanFiles:      孤儿文件清理（每天凌晨 4 点）
- cleanupTempFiles:        临时文件清理（每 6 小时）
- cleanupStuckPredEvalLogs: 回收预测/评估僵尸任务（每 60 秒）
- cleanupExpiredMessages:  清理过期消息（每天凌晨 4 点）
- sendScheduledAnnouncements: 发送定时公告（每分钟）
- expireOrders:            订单超时自动取消（每 5 分钟）
- completeExpiredOrders:   已支付订单到期归档（每天凌晨 3 点）
- expireUserCoupons:       用户优惠券过期处理（每天凌晨 4 点）
- autoRenew:               自动续费扣款（每天凌晨 8 点）
- resetMonthlyQuota:       会员月度配额重置（每月 1 日凌晨 0 点）
- processExpiredMembers:  会员过期降级（每天凌晨 2 点）
- retryFailedRefunds:     退款失败重试（每 30 分钟）
- sendExpireReminders:    会员到期预警（每天 09:00）
- refreshUnreadCountCache: 未读数缓存全量刷新（每小时）
- archiveInactiveConversations: AI 会话自动归档（每天凌晨 0 点）
- purgeDeletedConversations: AI 会话物理清理（软删超 30 天，每天凌晨 1:30）
- aiMemoryForget:         记忆遗忘归档（每天凌晨 3 点）
- aiMemoryReflection:     记忆反思整合（每天凌晨 4 点）
- aiMemoryMerge:          记忆合并去重（每天凌晨 5 点）
- purgeDeletedMemories:   记忆物理清理（软删超 30 天，每天凌晨 6 点）
- flushProviderKeyLastUsed: 供应商 API Key 最近使用信息批量刷库（每分钟）
- aiScheduleTrigger:      定时任务扫描触发（每分钟）
- aiScheduleRunCleanup:   定时任务执行历史清理（保留 30 天，每天凌晨 4 点）
- generateMonthlyBill:    月结账单生成（每月 1 日凌晨 0:30）
- clearVipGiftExpire:     VIP 赠送积分月末清零（每月最后一天 23:59）
- grantVipMonthlyGift:    VIP 按月赠送积分发放（每月 1 日凌晨 0 点）
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from pyxxl import JobHandler

from app.core.constants import (
    SYSTEM_USER_ID,
    TASK_CACHE_PREFIX,
    TASK_CANCEL_PREFIX,
    TASK_PROGRESS_PREFIX,
)
from app.database import get_db_session
from app.models.base import set_current_user_id
from app.repository.task_repository import task_repository

logger = logging.getLogger(__name__)

# 全局 handler 注册器（由 executor.py 导入并绑定到 PyxxlRunner）
xxl_handler = JobHandler()


@xxl_handler.register(name="cleanupExpiredTasks")
async def cleanup_expired_tasks() -> str:
    """
    清理过期任务

    - 删除 7 天前已完成/已取消的任务
    - 删除 30 天前所有已终止的任务（排除 pending/processing）
    - 精准清理对应的 Redis 缓存（仅删除已被数据库删除的任务 Key）

    CRON 建议: 0 0 2 * * ? （每天凌晨 2 点）
    """
    set_current_user_id(SYSTEM_USER_ID)
    try:
        now = datetime.now()
        seven_days_ago = now - timedelta(days=7)
        thirty_days_ago = now - timedelta(days=30)

        async with get_db_session() as db:
            # 先收集要删除的 task_id（用于后续 Redis 精准清理）
            terminated_task_ids = await task_repository.get_terminated_task_ids(db, seven_days_ago)

            # 删除 7 天前已完成/已取消的任务
            deleted_completed = await task_repository.delete_finished_before(
                db, seven_days_ago
            )

            # 删除 30 天前已终止的任务（排除 pending/processing，防止误删正在执行的任务）
            deleted_old = await task_repository.delete_terminated_before(
                db, thirty_days_ago
            )

        total = deleted_completed + deleted_old

        # 精准清理 Redis 缓存（仅删除已在 DB 中被删除的任务 Key）
        redis_deleted = await _cleanup_task_redis_keys(terminated_task_ids)

        msg = (
            f"过期任务清理完成: "
            f"7天前已完成/取消={deleted_completed}, "
            f"30天前已终止={deleted_old}, "
            f"总计删除={total}, "
            f"Redis缓存清理={redis_deleted}"
        )
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="cleanupStuckTasks")
async def cleanup_stuck_tasks() -> str:
    """
    回收僵死任务

    将超过 30 分钟仍处于 processing 状态的任务标记为 failed。
    将超过 24 小时仍处于 pending 状态的任务标记为 failed。
    这些任务可能由于进程崩溃、网络中断等原因未正常完成。

    CRON 建议: 0 0 * * * ? （每小时）
    """
    set_current_user_id(SYSTEM_USER_ID)
    try:
        now = datetime.now()
        processing_threshold = now - timedelta(minutes=30)
        pending_threshold = now - timedelta(hours=24)

        async with get_db_session() as db:
            # 回收 30 分钟未完成的 processing 任务（基于 started_at）
            recovered_processing = await task_repository.recover_stuck_processing(
                db, processing_threshold, now
            )

            # 回收 24 小时未启动的 pending 任务
            recovered_pending = await task_repository.recover_stuck_pending(
                db, pending_threshold, now
            )

        total = recovered_processing + recovered_pending

        msg = (
            f"僵死任务回收完成: "
            f"processing超时={recovered_processing}(>30min), "
            f"pending超时={recovered_pending}(>24h), "
            f"总计回收={total}"
        )
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="modelHealthCheck")
async def model_health_check() -> str:
    """
    模型健康检查

    检查 GPU 设备可用性和已加载模型的状态，
    异常时记录告警日志（后续可对接告警通知）。

    CRON 建议: 0 */30 * * * ? （每 30 分钟）
    """
    set_current_user_id(SYSTEM_USER_ID)
    try:
        issues: list[str] = []

        # 检查 GPU 可用性
        try:
            import torch

            if not torch.cuda.is_available():
                issues.append("CUDA 不可用")
            else:
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    used_pct = (total_mem - free_mem) / total_mem * 100
                    if used_pct > 95:
                        issues.append(f"GPU:{i} 显存使用率过高: {used_pct:.1f}%")
        except Exception as e:
            issues.append(f"GPU 检查异常: {e}")

        # 检查数据库连接
        try:
            from sqlalchemy import text

            async with get_db_session() as db:
                await db.execute(text("SELECT 1"))
        except Exception as e:
            issues.append(f"数据库连接异常: {e}")

        # 检查 Redis 连接
        try:
            from app.dependencies.redis import check_redis_health

            await check_redis_health()
        except Exception as e:
            issues.append(f"Redis 检查异常: {e}")

        if issues:
            msg = f"健康检查发现 {len(issues)} 个问题: {'; '.join(issues)}"
            logger.warning(msg)
        else:
            msg = "健康检查通过: GPU/DB/Redis 均正常"
            logger.debug(msg)

        return msg
    finally:
        set_current_user_id(None)


async def _cleanup_task_redis_keys(task_ids: list[str]) -> int:
    """
    精准清理 Redis 中指定任务的缓存 Key

    仅删除已经从数据库中被删除的任务对应的 Key，不影响正在执行的任务。

    Args:
        task_ids: 需要清理的任务 UUID 列表

    Returns:
        删除的 Key 数量
    """
    if not task_ids:
        return 0

    try:
        from app.dependencies.redis import get_redis_client

        redis = await get_redis_client()
        if redis is None:
            return 0

        # 精准删除：根据 task_id 构造 Key
        keys_to_delete = []
        for tid in task_ids:
            keys_to_delete.append(f"{TASK_CACHE_PREFIX}{tid}")
            keys_to_delete.append(f"{TASK_PROGRESS_PREFIX}{tid}")
            keys_to_delete.append(f"{TASK_CANCEL_PREFIX}{tid}")

        deleted = 0
        if keys_to_delete:
            deleted = await redis.delete(*keys_to_delete)

        if deleted > 0:
            logger.debug(
                f"已精准清理 {deleted} 个 Redis 任务缓存 Key（涉及 {len(task_ids)} 个任务）"
            )

        return deleted

    except Exception as e:
        logger.warning(f"Redis 任务缓存清理失败（不影响主流程）: {e}")
        return 0


# ==================== 文件清理定时任务 ====================


@xxl_handler.register(name="cleanupOrphanFiles")
async def cleanup_orphan_files() -> str:
    """
    孤儿文件清理

    清理存储中存在但数据库无对应记录的文件（通常由上传事务回滚导致）。
    仅清理超过阈值时间（默认 48h）的孤儿文件，避免误删正在上传中的文件。

    CRON 建议: 0 0 4 * * ? （每天凌晨 4 点）
    """
    set_current_user_id(SYSTEM_USER_ID)
    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        from app.config import settings
        from app.repository.file_repository import file_repository
        from app.service.storage.factory import get_storage_service

        bucket_name = settings.MINIO_BUCKET_NAME

        async with get_db_session() as db:
            # 获取数据库中所有文件的 object_name
            db_object_names = set(await file_repository.get_all_object_names(db))

        # 从存储中获取所有对象
        storage = get_storage_service()
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="file-cleanup")
        loop = asyncio.get_running_loop()

        try:
            storage_objects = await loop.run_in_executor(
                executor, lambda: storage.list_objects(bucket_name, prefix="upload/")
            )
        except Exception as e:
            msg = f"孤儿文件清理失败: 无法列出存储对象: {e}"
            logger.error(msg)
            return msg

        # 找出孤儿文件（存储中有但 DB 没有的）
        orphan_objects = [obj for obj in storage_objects if obj not in db_object_names]

        if not orphan_objects:
            msg = "孤儿文件清理完成: 未发现孤儿文件"
            logger.debug(msg)
            return msg

        # 删除孤儿文件
        deleted = 0
        failed = 0
        for obj_name in orphan_objects:

            def _delete(name=obj_name):
                try:
                    storage.delete(bucket_name, name)
                    return True
                except Exception as e:
                    logger.warning(f"孤儿文件删除失败 [{name}]: {e}")
                    return False

            try:
                ok = await loop.run_in_executor(executor, _delete)
                if ok:
                    deleted += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        executor.shutdown(wait=False)

        msg = f"孤儿文件清理完成: 发现={len(orphan_objects)}, 已删除={deleted}, 失败={failed}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="cleanupTempFiles")
async def cleanup_temp_files() -> str:
    """
    临时文件清理

    清理超过阈值时间（默认 24h）的临时目录中的文件。
    临时文件由文件处理任务产生，正常情况下任务完成后会清理，
    但异常中断时可能残留。

    CRON 建议: 0 0 */6 * * ? （每 6 小时）
    """
    set_current_user_id(SYSTEM_USER_ID)
    try:
        import os
        import time

        from app.config import settings

        temp_dir = settings.TEMP_DIR_RESOLVED
        threshold_hours = settings.FILE_TEMP_CLEANUP_HOURS
        threshold_seconds = threshold_hours * 3600
        now = time.time()

        if not os.path.exists(temp_dir):
            msg = f"临时文件清理: 目录不存在 {temp_dir}"
            logger.debug(msg)
            return msg

        deleted = 0
        failed = 0

        for root, _dirs, files in os.walk(temp_dir, topdown=False):
            for f in files:
                filepath = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(filepath)
                    if now - mtime > threshold_seconds:
                        os.unlink(filepath)
                        deleted += 1
                except Exception as e:
                    logger.warning(f"临时文件删除失败 [{filepath}]: {e}")
                    failed += 1

            # 尝试删除空目录（不删除 temp_dir 本身）
            if root != temp_dir:
                try:
                    if not os.listdir(root):
                        os.rmdir(root)
                except OSError as e:
                    logger.warning(f"删除空目录失败 [{root}]: {e}")

        msg = (
            f"临时文件清理完成: "
            f"目录={temp_dir}, "
            f"阈值={threshold_hours}h, "
            f"已删除={deleted}, "
            f"失败={failed}"
        )
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="cleanupStuckPredEvalLogs")
async def cleanup_stuck_pred_eval_logs() -> str:
    """
    回收预测/评估僵尸任务

    扫描 status=1(处理中) AND update_time < NOW() - 10 MINUTE 的记录，
    标记为 status=3(失败), error_message='任务执行超时，服务可能已重启'。
    与 Java PredEvalLogCleanupJob、Go cleanupStuckPredEvalLogs 对齐。

    CRON 建议: 0 * * * * ? （每 60 秒）
    """
    from app.repository.pred_eval_log_repository import (
        eval_log_repository,
        pred_log_repository,
    )

    set_current_user_id(SYSTEM_USER_ID)
    try:
        threshold = datetime.now() - timedelta(minutes=10)

        async with get_db_session() as db:
            pred_count = await pred_log_repository.mark_stuck_as_failed(
                db=db,
                threshold=threshold,
            )
            eval_count = await eval_log_repository.mark_stuck_as_failed(
                db=db,
                threshold=threshold,
            )

        if pred_count > 0 or eval_count > 0:
            msg = f"回收预测/评估僵尸任务: pred={pred_count}, eval={eval_count}"
            logger.warning(msg)
        else:
            msg = "回收预测/评估僵尸任务: 无"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


# ==================== 消息通知定时任务 ====================


@xxl_handler.register(name="cleanupExpiredMessages")
async def cleanup_expired_messages() -> str:
    """
    清理过期消息

    删除 expires_at < NOW() 的消息记录（物理删除）。
    分批处理，每批 500 条，避免长时间锁表。
    与 Java MessageCleanupJob、Go cleanupExpiredMessages 对齐。

    CRON 建议: 0 0 4 * * ? （每天凌晨 4 点）
    """
    from app.repository.message_repository import message_repository

    set_current_user_id(SYSTEM_USER_ID)
    try:
        now = datetime.now()
        async with get_db_session() as db:
            total = await message_repository.delete_expired(db, now, batch_size=500)

        msg = f"过期消息清理完成: 已删除={total}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="sendScheduledAnnouncements")
async def send_scheduled_announcements() -> str:
    """
    发送定时公告

    扫描 status=2(待发送) AND send_time <= NOW() 的公告，
    逐条调用 AnnouncementService.send 完成投递。
    与 Java AnnouncementScheduleJob、Go sendScheduledAnnouncements 对齐。

    CRON 建议: 0 * * * * ? （每分钟）
    """
    from app.repository.announcement_repository import announcement_repository
    from app.service.announcement_service import AnnouncementService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        now = datetime.now()
        sent_total = 0
        failed = 0

        async with get_db_session() as db:
            pending = await announcement_repository.get_scheduled_pending(db, now)

        for announcement in pending:
            try:
                async with get_db_session() as db:
                    await AnnouncementService.send(db, announcement.id)
                    sent_total += 1
            except Exception as e:
                failed += 1
                logger.warning(f"定时公告发送失败 id={announcement.id}: {e}")

        if sent_total > 0 or failed > 0:
            msg = f"定时公告发送: 成功={sent_total}, 失败={failed}"
            logger.debug(msg)
        else:
            msg = "定时公告发送: 无待发送公告"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


# ==================== 订单与套餐定时任务 ====================


@xxl_handler.register(name="expireOrders")
async def expire_orders() -> str:
    """
    订单超时自动取消

    扫描 status=1(待支付) AND expire_time < NOW() 的订单，
    释放已锁定优惠券，更新状态为 cancelled，cancel_reason 标记为超时。
    与 Java OrderExpireJob、Go expireOrders 对齐。

    CRON 建议: 0 0/5 * * * ? （每 5 分钟）
    """
    from app.service.order_service import OrderService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            count = await OrderService.expire_orders(db)

        if count > 0:
            msg = f"订单超时取消: 已取消={count}"
            logger.debug(msg)
        else:
            msg = "订单超时取消: 无"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="completeExpiredOrders")
async def complete_expired_orders() -> str:
    """
    已支付订单到期归档

    扫描 status=2(已支付) AND package_expire_time < NOW() 的订单，
    更新状态为 completed（归档）。
    与 Java OrderCompleteJob、Go completeExpiredOrders 对齐。

    CRON 建议: 0 0 3 * * ? （每天凌晨 3 点）
    """
    from app.service.order_service import OrderService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            count = await OrderService.complete_expired_orders(db)

        if count > 0:
            msg = f"订单到期归档: 已归档={count}"
            logger.debug(msg)
        else:
            msg = "订单到期归档: 无"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="expireUserCoupons")
async def expire_user_coupons() -> str:
    """
    用户优惠券过期处理

    扫描 sys_user_coupon WHERE expire_time < NOW() AND status = 1(未使用)，
    批量更新 status=3(已过期)。
    与 Java CouponExpireJob、Go expireUserCoupons 对齐。

    CRON 建议: 0 0 4 * * ? （每天凌晨 4 点）
    """
    from app.service.coupon_service import CouponService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            count = await CouponService.expire_user_coupons(db)

        if count > 0:
            msg = f"用户优惠券过期处理: 已过期={count}"
            logger.debug(msg)
        else:
            msg = "用户优惠券过期处理: 无"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="autoRenew")
async def auto_renew_task() -> str:
    """
    自动续费扣款

    扫描 sys_auto_renew WHERE status=1(生效中) AND next_renew_time <= NOW()，
    按支付方式发起代扣：
      - balance: 直接扣减余额并完成订单
      - wechat/alipay: 调用支付渠道 unified_order 下单
    成功后更新 next_renew_time，失败累计 retry_count，超过 AUTO_RENEW_RETRY_MAX 后停用。
    与 Java AutoRenewJob、Go autoRenew 对齐。

    CRON 建议: 0 0 8 * * ? （每天凌晨 8 点）
    """
    from app.service.order_service import OrderService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            success_count = await OrderService.execute_renewal(db)

        if success_count > 0:
            msg = f"自动续费扣款完成: 成功={success_count}"
            logger.debug(msg)
        else:
            msg = "自动续费扣款: 无待处理配置"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="resetMonthlyQuota")
async def reset_monthly_quota() -> str:
    """
    会员月度配额重置

    每月 1 日扫描 sys_member WHERE quota_reset_month != 当前月份，
    按当前等级权益重置 monthly_dehaze_quota/monthly_evaluate_quota 字段，
    并将 used 字段清零，quota_reset_month 更新为当前月份。
    与 Java MemberMonthlyQuotaResetJob、Go resetMonthlyQuota 对齐。

    CRON 建议: 0 0 0 1 * ? （每月 1 日凌晨 0 点）
    """
    from app.service.member_service import MemberService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            count = await MemberService.reset_monthly_quota(db)

        if count > 0:
            msg = f"会员月度配额重置完成: 已重置={count}"
            logger.debug(msg)
        else:
            msg = "会员月度配额重置: 无待处理记录"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="processExpiredMembers")
async def process_expired_members() -> str:
    """
    会员过期降级处理

    扫描 sys_member WHERE expire_time < NOW() AND level_source != 'growth' 的会员，
    按成长值重算等级、置 level_source='growth'、清空 expire_time、刷新权益。
    与 Java MemberExpireJob、Go processExpiredMembers 对齐。

    CRON 建议: 0 0 2 * * ? （每天凌晨 2 点）
    """
    from app.service.member_service import MemberService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            count = await MemberService.process_expired_members(db)

        if count > 0:
            msg = f"会员过期降级处理完成: 已处理={count}"
            logger.debug(msg)
        else:
            msg = "会员过期降级处理: 无待处理记录"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="retryFailedRefunds")
async def retry_failed_refunds() -> str:
    """
    退款失败重试

    扫描 sys_refund_record WHERE status=3(退款失败) AND retry_count < 3 的记录，
    重新调用渠道退款接口，重试次数达上限则标记为最终失败。
    与 Java RefundRetryJob、Go retryFailedRefunds 对齐。

    CRON 建议: 0 0/30 * * * ? （每 30 分钟）
    """
    from app.service.order_service import OrderService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            count = await OrderService.retry_failed_refunds(db)

        if count > 0:
            msg = f"退款失败重试完成: 已处理={count}"
            logger.debug(msg)
        else:
            msg = "退款失败重试: 无待处理记录"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="sendExpireReminders")
async def send_expire_reminders() -> str:
    """
    会员到期预警

    扫描 expire_time 在未来 7/3/1 天的会员，推送续费提醒站内信：
      - 7 天：普通提醒
      - 3 天：含降级权益对比
      - 1 天：最后提醒
    与 Java MemberExpireReminderJob、Go sendExpireReminders 对齐。

    CRON 建议: 0 0 9 * * ? （每天 09:00）
    """
    from app.service.member_service import MemberService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            count = await MemberService.send_expire_reminders(db)

        if count > 0:
            msg = f"会员到期预警完成: 已发送={count}"
            logger.debug(msg)
        else:
            msg = "会员到期预警: 无待处理记录"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="refreshUnreadCountCache")
async def refresh_unread_count_cache() -> str:
    """
    未读数缓存全量刷新

    每小时扫描所有活跃用户，重新计算未读数并刷新 Redis 缓存。
    与 Java UnreadCountRefreshJob、Go refreshUnreadCountCache 对齐。

    CRON 建议: 0 0 * * * ? （每小时整点）
    """
    from app.service.message_service import MessageService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            count = await MessageService.refresh_unread_count_cache(db)

        if count > 0:
            msg = f"未读数缓存刷新完成: 已刷新={count}"
            logger.debug(msg)
        else:
            msg = "未读数缓存刷新: 无活跃用户"
            logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


# ==================== AI 记忆整理定时任务 ====================


@xxl_handler.register(name="aiMemoryForget")
async def ai_memory_forget() -> str:
    """
    记忆遗忘归档

    基于 Ebbinghaus 遗忘曲线计算记忆衰减（priority = importance × exp(-Δt/half_life)），
    归档 priority < 阈值的记忆（archived=1），归档后不再注入对话但保留记录。
    与 Java MemoryForgetJob、Go aiMemoryForget 对齐。

    CRON 建议: 0 0 3 * * ? （每天凌晨 3 点）
    """
    from app.config import settings
    from app.repository.ai_memory_repository import ai_memory_repository

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            count = await ai_memory_repository.archive_forgotten(
                db,
                threshold=settings.AI_MEMORY_FORGET_THRESHOLD,
                half_life_days=settings.AI_MEMORY_HALF_LIFE_DAYS,
            )
        msg = f"记忆遗忘归档完成: 已归档={count}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="aiMemoryReflection")
async def ai_memory_reflection() -> str:
    """
    记忆反思整合

    遍历所有拥有活跃记忆的用户，回顾其近 7 天情景记忆，
    调用 LLM 分析规律并生成更高层次的抽象洞察（source=reflection）。
    与 Java MemoryReflectionJob、Go aiMemoryReflection 对齐。

    CRON 建议: 0 0 4 * * ? （每天凌晨 4 点）
    """
    from app.config import settings
    from app.repository.ai_memory_repository import ai_memory_repository
    from app.service.ai.memory_extraction import reflect_and_consolidate

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            user_ids = await ai_memory_repository.get_active_user_ids(db)
        total = 0
        for user_id in user_ids:
            try:
                async with get_db_session() as db:
                    total += await reflect_and_consolidate(db, user_id, settings.AI_DEFAULT_MODEL)
            except Exception as e:
                logger.warning("记忆反思失败 user_id=%s: %s", user_id, e)
        msg = f"记忆反思整合完成: 处理用户={len(user_ids)}, 新增洞察={total}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="aiMemoryMerge")
async def ai_memory_merge() -> str:
    """
    记忆合并去重

    遍历所有拥有活跃记忆的用户，检测同类型语义重复记忆（相似度 > 0.9），
    调用 LLM 合并为更完整的单一条目，旧记忆软删除。
    与 Java MemoryMergeJob、Go aiMemoryMerge 对齐。

    CRON 建议: 0 0 5 * * ? （每天凌晨 5 点）
    """
    from app.config import settings
    from app.repository.ai_memory_repository import ai_memory_repository
    from app.service.ai.memory_extraction import merge_duplicates

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            user_ids = await ai_memory_repository.get_active_user_ids(db)
        total = 0
        for user_id in user_ids:
            try:
                async with get_db_session() as db:
                    total += await merge_duplicates(db, user_id, settings.AI_DEFAULT_MODEL)
            except Exception as e:
                logger.warning("记忆合并失败 user_id=%s: %s", user_id, e)
        msg = f"记忆合并去重完成: 处理用户={len(user_ids)}, 合并={total}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="purgeDeletedMemories")
async def purge_deleted_memories() -> str:
    """
    记忆物理清理（软删超过 30 天）

    清理 deleted=1 且 delete_time < NOW() - 30 天的记忆（物理 DELETE），
    与软删恢复窗口（30 天）对齐，超期记忆不再可恢复。
    与 Java MemoryPurgeJob、Go purgeDeletedMemories 对齐。

    CRON 建议: 0 0 6 * * ? （每天凌晨 6 点）
    """
    from app.models.entity.sys_ai_memory import MEMORY_RECOVERY_WINDOW_DAYS
    from app.repository.ai_memory_repository import ai_memory_repository

    set_current_user_id(SYSTEM_USER_ID)
    try:
        before_date = datetime.now() - timedelta(days=MEMORY_RECOVERY_WINDOW_DAYS)
        async with get_db_session() as db:
            ids = await ai_memory_repository.list_deleted_for_purge(db, before_date)
            if ids:
                await ai_memory_repository.delete_by_ids(db, ids)
        msg = f"记忆物理清理完成: 已清理={len(ids)}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="archiveInactiveConversations")
async def archive_inactive_conversations() -> str:
    """
    AI 会话自动归档

    扫描 status=1(活跃) 且 last_message_at < NOW() - 30 天的会话，
    更新 status=2(已归档)，并清除相关用户的会话列表缓存。
    与 Java ArchiveInactiveConversationJob、Go archiveInactiveConversations 对齐。

    CRON 建议: 0 0 0 * * ? （每天凌晨 0 点）
    """
    from app.dependencies.redis import get_redis_client
    from app.repository.ai_conversation_repository import ai_conversation_repository

    set_current_user_id(SYSTEM_USER_ID)
    try:
        before_date = datetime.now() - timedelta(days=30)
        async with get_db_session() as db:
            archived = await ai_conversation_repository.archive_inactive(db, before_date)

        user_ids = {uid for _, uid in archived}
        if user_ids:
            redis = await get_redis_client()
            for uid in user_ids:
                await redis.delete(f"ai:conv:list:{uid}")

        msg = f"AI 会话自动归档完成: 已归档={len(archived)}, 涉及用户={len(user_ids)}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="purgeDeletedConversations")
async def purge_deleted_conversations() -> str:
    """
    AI 会话物理清理（软删超过 30 天）

    清理 deleted=1 且 delete_time < NOW() - 30 天的会话（物理 DELETE），
    级联物理删除其消息记录（sys_ai_message）。与软删恢复窗口（30 天）对齐。

    CRON 建议: 0 30 1 * * ? （每天凌晨 1:30）
    """
    from app.repository.ai_conversation_repository import ai_conversation_repository
    from app.repository.ai_message_repository import ai_message_repository

    set_current_user_id(SYSTEM_USER_ID)
    try:
        before_date = datetime.now() - timedelta(days=30)
        async with get_db_session() as db:
            conv_ids = await ai_conversation_repository.list_soft_deleted_before(db, before_date)
            if conv_ids:
                await ai_message_repository.delete_by_conversations(db, conv_ids)
                await ai_conversation_repository.delete_by_ids(db, conv_ids)
                await db.commit()

        msg = f"AI 会话物理清理完成: 已清理会话={len(conv_ids)}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


# ==================== AI 供应商管理定时任务 ====================


@xxl_handler.register(name="flushProviderKeyLastUsed")
async def flush_provider_key_last_used() -> str:
    """
    供应商 API Key 最近使用信息批量刷库

    每分钟读取 Redis 缓冲（ai:provider_key:{id}:last_used），批量调用
    batch_update_last_used 刷新 last_used_at/last_used_by，并清除已落库的缓冲。
    与 Java ProviderKeyLastUsedFlushJob、Go flushProviderKeyLastUsed 对齐。

    CRON 建议: 0 * * * * ? （每分钟）
    """
    from app.dependencies.redis import get_redis_client
    from app.repository.ai_provider_key_repository import ai_provider_key_repository

    set_current_user_id(SYSTEM_USER_ID)
    try:
        import json as _json

        redis = await get_redis_client()
        if redis is None:
            return "供应商Key最近使用刷库: Redis 不可用"

        keys = []
        async for key in redis.scan_iter(match="ai:provider_key:*:last_used"):
            keys.append(key)

        updates: list[tuple[int, datetime, int]] = []
        for key in keys:
            # key 形如 ai:provider_key:{key_id}:last_used
            try:
                key_id = int(key.split(":")[2])
                raw = await redis.get(key)
                if not raw:
                    continue
                data = _json.loads(raw)
                used_at = datetime.fromisoformat(data["last_used_at"])
                updates.append((key_id, used_at, data.get("last_used_by")))
            except (ValueError, KeyError, TypeError):
                logger.warning("跳过非法 last_used 缓冲: %s", key)

        if not updates:
            return "供应商Key最近使用刷库: 无待刷缓冲"

        async with get_db_session() as db:
            await ai_provider_key_repository.batch_update_last_used(db, updates)
            await db.commit()

        # 落库成功后清除缓冲
        await redis.delete(*keys)
        msg = f"供应商Key最近使用刷库完成: 已刷 {len(updates)} 条"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


# ==================== AI 计费管理定时任务 ====================


@xxl_handler.register(name="generateMonthlyBill")
async def generate_monthly_bill() -> str:
    """
    月结账单生成

    每月 1 日扫描上月有 AI 消耗/流水记录的用户，生成上月账单并缓存到 Redis。
    与 Java BillGeneratorTask、Go generateMonthlyBill 对齐。

    CRON 建议: 0 30 0 1 * ? （每月 1 日凌晨 00:30）
    """
    from datetime import date

    from app.repository.ai_billing_repository import ai_billing_repository
    from app.repository.ai_credit_log_repository import ai_credit_log_repository
    from app.service.billing.bill_service import BillService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        # 上月账期
        today = date.today()
        first_of_this_month = today.replace(day=1)
        if first_of_this_month.month == 1:
            last_month = first_of_this_month.replace(year=first_of_this_month.year - 1, month=12)
        else:
            last_month = first_of_this_month.replace(month=first_of_this_month.month - 1)
        month = last_month.strftime("%Y-%m")
        start = datetime(last_month.year, last_month.month, 1)
        end = start + timedelta(days=32)
        end = end.replace(day=1) - timedelta(seconds=1)

        async with get_db_session() as db:
            billing_users = await ai_billing_repository.distinct_user_ids(db, start, end)
            log_users = await ai_credit_log_repository.distinct_user_ids_by_source(
                db, "consume", start, end
            )
        user_ids = sorted(set(billing_users) | set(log_users))

        generated = 0
        for user_id in user_ids:
            try:
                async with get_db_session() as db:
                    await BillService.generate_monthly_bill(db, user_id, month)
                    generated += 1
            except Exception as e:
                logger.warning("月结账单生成失败 user_id=%s month=%s: %s", user_id, month, e)

        msg = f"月结账单生成完成: 账期={month}, 用户数={len(user_ids)}, 成功={generated}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


# ==================== AI 定时调度定时任务 ====================


@xxl_handler.register(name="aiScheduleTrigger")
async def ai_schedule_trigger() -> str:
    """
    定时任务扫描触发（无人值守执行）

    扫描到期（enabled=1 AND status=1 AND next_trigger_time <= NOW）的 AI 定时任务，
    逐条调用 ScheduleExecutor.scan_and_trigger 走完整执行链路（幂等防重入/并发控制/
    配额保护/失败重试/连续失败熔断/执行历史）。与 Java AIScheduleTriggerJob、
    Go aiScheduleTrigger 对齐。

    CRON 建议: 0 * * * * ? （每分钟）
    """
    from app.dependencies.redis import get_redis_client
    from app.service.ai.ai_schedule_executor import schedule_executor

    set_current_user_id(SYSTEM_USER_ID)
    try:
        redis = await get_redis_client()
        if redis is None:
            return "定时任务扫描触发: Redis 不可用"

        async with get_db_session() as db:
            summary = await schedule_executor.scan_and_trigger(db, redis)
        msg = (
            f"定时任务扫描触发完成: 扫描={summary['scanned']}, "
            f"触发={summary['triggered']}, 跳过={summary['skipped']}, 失败={summary['failed']}"
        )
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="aiScheduleRunCleanup")
async def ai_schedule_run_cleanup() -> str:
    """
    定时任务执行历史清理（保留 30 天）

    物理清理 create_time < NOW() - 30 天的执行历史记录（sys_ai_schedule_run），
    含成功/失败/跳过与熔断记录一并归档删除。与 Java AIScheduleRunCleanupJob、
    Go aiScheduleRunCleanup 对齐。

    CRON 建议: 0 0 4 * * ? （每天凌晨 4 点）
    """
    from app.repository.ai_schedule_run_repository import ai_schedule_run_repository

    set_current_user_id(SYSTEM_USER_ID)
    try:
        before = datetime.now() - timedelta(days=30)
        async with get_db_session() as db:
            deleted = await ai_schedule_run_repository.cleanup_before(db, before)
            await db.commit()

        msg = f"定时任务执行历史清理完成: 已清理={deleted}（30 天前记录）"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="clearVipGiftExpire")
async def clear_vip_gift_expire() -> str:
    """
    VIP 赠送积分月末清零

    每月最后一天 23:59 统计当月 VIP 赠送（source=vip_gift）未用部分，
    从余额中清零并记录流水（source=vip_gift_expire）。
    与 Java VipGiftClearJob、Go clearVipGiftExpire 对齐。

    CRON 建议: 0 59 23 L * ? （每月最后一天 23:59）
    """
    from app.repository.ai_credit_log_repository import ai_credit_log_repository
    from app.service.billing.balance_service import balance_service

    set_current_user_id(SYSTEM_USER_ID)
    try:
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_end = now.replace(day=28, hour=23, minute=59, second=59, microsecond=999999)
        month_end = (month_end + timedelta(days=4)).replace(day=1) - timedelta(seconds=1)

        async with get_db_session() as db:
            gift_users = await ai_credit_log_repository.distinct_user_ids_by_source(
                db, "vip_gift", month_start, month_end
            )

        cleared = 0
        for user_id in gift_users:
            try:
                async with get_db_session() as db:
                    # 当月赠送总额
                    by_source = await ai_credit_log_repository.sum_amount_by_user_and_source(
                        db, user_id, month_start, month_end
                    )
                    gift_total = int(by_source.get("vip_gift", 0))
                    balance = int(await balance_service.get_balance(db, user_id))
                    expire_amount = min(gift_total, balance)
                    if expire_amount <= 0:
                        continue
                    # 扣减（amount <= 余额不会触发欠费标记）
                    await balance_service.deduct(db, user_id, expire_amount)
                    balance_after = int(await balance_service.get_balance(db, user_id))
                    await ai_credit_log_repository.create_log(
                        db,
                        user_id=user_id,
                        source="vip_gift_expire",
                        amount=-expire_amount,
                        balance_after=balance_after,
                        reason="VIP 赠送积分月末清零",
                    )
                    cleared += 1
            except Exception as e:
                logger.warning("VIP赠送清零失败 user_id=%s: %s", user_id, e)

        msg = f"VIP赠送积分月末清零完成: 涉及用户={len(gift_users)}, 清零={cleared}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)


@xxl_handler.register(name="grantVipMonthlyGift")
async def grant_vip_monthly_gift() -> str:
    """
    VIP 按月赠送积分发放

    每月 1 日扫描配置了 vip_gift_credits（>0）的启用等级，
    逐等级分页扫描活跃会员并按等级额度发放赠送积分（source=vip_gift）。
    与后端实现 §5.3 / §9 对齐（Java/Go 对齐时任务名 grantVipMonthlyGift）。

    CRON 建议: 0 0 0 1 * ? （每月 1 日凌晨 0 点）
    """
    from app.repository.member_benefit_repository import member_benefit_repository
    from app.repository.member_repository import member_repository
    from app.service.billing.recharge_service import RechargeService

    set_current_user_id(SYSTEM_USER_ID)
    try:
        async with get_db_session() as db:
            benefits = await member_benefit_repository.list_all(db)

        # 待发放等级：启用且配置了赠送额度
        gift_levels = [
            b for b in benefits if b.status == 1 and (b.vip_gift_credits or 0) > 0
        ]
        granted = 0
        for benefit in gift_levels:
            amount = int(benefit.vip_gift_credits)
            offset = 0
            while True:
                async with get_db_session() as db:
                    members = await member_repository.list_active_by_level(
                        db, benefit.level_code, offset=offset, limit=500
                    )
                if not members:
                    break
                for member in members:
                    try:
                        async with get_db_session() as db:
                            await RechargeService.grant_vip_monthly_gift(
                                db, member.user_id, amount
                            )
                            granted += 1
                    except Exception as e:
                        logger.warning(
                            "VIP月度赠送失败 user_id=%s level=%s: %s",
                            member.user_id,
                            benefit.level_code,
                            e,
                        )
                offset += len(members)

        msg = f"VIP月度赠送完成: 等级数={len(gift_levels)}, 发放用户数={granted}"
        logger.debug(msg)
        return msg
    finally:
        set_current_user_id(None)
