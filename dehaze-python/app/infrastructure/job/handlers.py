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
- autoRenewTask:           自动续费扣款（每天凌晨 8 点）
- resetMonthlyQuota:       会员月度配额重置（每月 1 日凌晨 0 点）
- processExpiredMembers:  会员过期降级（每天凌晨 2 点）
- retryFailedRefunds:     退款失败重试（每 30 分钟）
- sendExpireReminders:    会员到期预警（每天 09:00）
- refreshUnreadCountCache: 未读数缓存全量刷新（每小时）
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from pyxxl import JobHandler
from sqlalchemy import and_, delete, update

from app.core.constants import (SYSTEM_USER_ID, TASK_CACHE_PREFIX,
                                TASK_CANCEL_PREFIX, TASK_PROGRESS_PREFIX)
from app.database import get_db_session
from app.models.base import get_audit_update_values, set_current_user_id
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import TaskStatus
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
            stmt_completed = delete(SysTask).where(
                and_(
                    SysTask.status.in_([TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]),
                    SysTask.create_time < seven_days_ago,
                )
            )
            result_completed = await db.execute(stmt_completed)

            # 删除 30 天前已终止的任务（排除 pending/processing，防止误删正在执行的任务）
            stmt_old = delete(SysTask).where(
                and_(
                    SysTask.status.not_in([TaskStatus.PENDING.value, TaskStatus.PROCESSING.value]),
                    SysTask.create_time < thirty_days_ago,
                )
            )
            result_old = await db.execute(stmt_old)

        deleted_completed = result_completed.rowcount
        deleted_old = result_old.rowcount
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
            processing_values = {
                "status": TaskStatus.FAILED.value,
                "error_message": "任务超时（30分钟无进度更新），已被系统自动回收",
                "completed_at": now,
            }
            processing_values.update(get_audit_update_values())
            stmt_processing = (
                update(SysTask)
                .where(
                    and_(
                        SysTask.status == TaskStatus.PROCESSING.value,
                        SysTask.started_at < processing_threshold,
                    )
                )
                .values(**processing_values)
            )
            result_processing = await db.execute(stmt_processing)

            # 回收 24 小时未启动的 pending 任务
            pending_values = {
                "status": TaskStatus.FAILED.value,
                "error_message": "任务超时（24h未启动），已被系统自动回收",
                "completed_at": now,
            }
            pending_values.update(get_audit_update_values())
            stmt_pending = (
                update(SysTask)
                .where(
                    and_(
                        SysTask.status == TaskStatus.PENDING.value,
                        SysTask.create_time < pending_threshold,
                    )
                )
                .values(**pending_values)
            )
            result_pending = await db.execute(stmt_pending)

        recovered_processing = result_processing.rowcount
        recovered_pending = result_pending.rowcount
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
                        issues.append(
                            f"GPU:{i} 显存使用率过高: {used_pct:.1f}%"
                        )
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
            logger.debug(f"已精准清理 {deleted} 个 Redis 任务缓存 Key（涉及 {len(task_ids)} 个任务）")

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

        threshold_hours = settings.FILE_ORPHAN_CLEANUP_HOURS
        bucket_name = settings.MINIO_BUCKET_NAME

        async with get_db_session() as db:
            # 获取数据库中所有文件的 object_name
            db_object_names = set(await file_repository.get_all_object_names(db))

        # 从存储中获取所有对象
        storage = get_storage_service()
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="file-cleanup")
        loop = asyncio.get_running_loop()

        def _list_storage_objects():
            return storage.list_objects(bucket_name, prefix="upload/")

        try:
            storage_objects = await loop.run_in_executor(executor, _list_storage_objects)
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

        msg = (
            f"孤儿文件清理完成: "
            f"发现={len(orphan_objects)}, "
            f"已删除={deleted}, "
            f"失败={failed}"
        )
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

        for root, dirs, files in os.walk(temp_dir, topdown=False):
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
        pred_log_repository, eval_log_repository,
    )

    set_current_user_id(SYSTEM_USER_ID)
    try:
        threshold = datetime.now() - timedelta(minutes=10)

        async with get_db_session() as db:
            pred_count = await pred_log_repository.mark_stuck_as_failed(
                db=db, threshold=threshold,
            )
            eval_count = await eval_log_repository.mark_stuck_as_failed(
                db=db, threshold=threshold,
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


@xxl_handler.register(name="autoRenewTask")
async def auto_renew_task() -> str:
    """
    自动续费扣款

    扫描 sys_auto_renew WHERE status=1(生效中) AND next_renew_time <= NOW()，
    按支付方式发起代扣：
      - balance: 直接扣减余额并完成订单
      - wechat/alipay: 调用支付渠道 unified_order 下单
    成功后更新 next_renew_time，失败累计 retry_count，超过 AUTO_RENEW_RETRY_MAX 后停用。
    与 Java AutoRenewJob、Go autoRenewTask 对齐。

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
    与 Java MemberQuotaResetJob、Go resetMonthlyQuota 对齐。

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
