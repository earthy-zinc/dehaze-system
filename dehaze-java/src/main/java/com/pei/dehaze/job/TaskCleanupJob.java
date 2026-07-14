package com.pei.dehaze.job;

import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.security.util.SystemSecurityContext;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 任务定时清理任务
 * 对齐 Python cleanupExpiredTasks / cleanupStuckTasks 的清理规则：
 * - 每天凌晨2点：7天前 COMPLETED/CANCELLED 物理删除 + 30天前所有非 PENDING/PROCESSING 物理删除
 * - 每小时：PROCESSING(startedAt<30min) / PENDING(createTime<24h) 标记 FAILED
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Slf4j
@Component
public class TaskCleanupJob {

    @Resource
    private SysTaskMapper sysTaskMapper;

    @Resource
    private RedisTemplate<String, Object> redisTemplate;

    /**
     * 每天凌晨2点执行清理任务
     * 对齐 Python cleanupExpiredTasks：
     * 1. 7天前 COMPLETED/CANCELLED 任务物理删除（用 createTime 判定，避免 completed_at 为 NULL 漏清理）
     * 2. 30天前所有非 PENDING/PROCESSING 任务物理删除（排除正在执行的任务）
     */
    @Scheduled(cron = "0 0 2 * * ?")
    public void cleanupExpiredTasks() {
        SystemSecurityContext.setSystemContext();
        try {
            log.info("开始清理过期导出任务...");

            physicalDeleteTasks(
                    new LambdaQueryWrapper<SysTask>()
                            .in(SysTask::getStatus,
                                    TaskConstants.STATUS_COMPLETED,
                                    TaskConstants.STATUS_CANCELLED)
                            .lt(SysTask::getCreateTime, LocalDateTime.now().minusDays(7)),
                    "7天前已完成/取消任务"
            );

            physicalDeleteTasks(
                    new LambdaQueryWrapper<SysTask>()
                            .notIn(SysTask::getStatus,
                                    TaskConstants.STATUS_PENDING,
                                    TaskConstants.STATUS_PROCESSING)
                            .lt(SysTask::getCreateTime, LocalDateTime.now().minusDays(30)),
                    "30天前已终止任务"
            );

            log.info("清理过期导出任务完成");
        } finally {
            SystemSecurityContext.clearContext();
        }
    }

    /**
     * 每小时执行一次，清理僵死任务
     * 对齐 Python cleanupStuckTasks：
     * 1. PROCESSING 且 startedAt < 30分钟 → FAILED（基于开始时间判定，非 createTime）
     * 2. PENDING 且 createTime < 24小时 → FAILED
     */
    @Scheduled(cron = "0 0 * * * ?")
    public void cleanupStuckTasks() {
        SystemSecurityContext.setSystemContext();
        try {
            log.info("开始清理僵死任务...");

            markStuckTasksAsFailed(
                    new LambdaQueryWrapper<SysTask>()
                            .eq(SysTask::getStatus, TaskConstants.STATUS_PROCESSING)
                            .lt(SysTask::getStartedAt, LocalDateTime.now().minusMinutes(30)),
                    "任务超时（30分钟无进度更新），已被系统自动回收"
            );

            markStuckTasksAsFailed(
                    new LambdaQueryWrapper<SysTask>()
                            .eq(SysTask::getStatus, TaskConstants.STATUS_PENDING)
                            .lt(SysTask::getCreateTime, LocalDateTime.now().minusDays(1)),
                    "任务超时（24h未启动），已被系统自动回收"
            );

            log.info("清理僵死任务完成");
        } finally {
            SystemSecurityContext.clearContext();
        }
    }

    /**
     * 物理删除任务记录并清除 Redis 缓存
     */
    private void physicalDeleteTasks(LambdaQueryWrapper<SysTask> wrapper, String logLabel) {
        try {
            List<SysTask> tasks = sysTaskMapper.selectList(wrapper);
            if (tasks.isEmpty()) {
                return;
            }

            List<Long> ids = tasks.stream()
                    .map(SysTask::getId)
                    .collect(Collectors.toList());
            sysTaskMapper.deleteBatchIds(ids);

            for (SysTask task : tasks) {
                if (StrUtil.isNotBlank(task.getTaskId())) {
                    redisTemplate.delete(TaskConstants.TASK_CACHE_PREFIX + task.getTaskId());
                }
            }
            log.info("清理{}: 共清理{}条记录", logLabel, tasks.size());
        } catch (Exception e) {
            log.error("清理{}失败", logLabel, e);
        }
    }

    /**
     * 将僵死任务标记为 FAILED 并清除缓存（evict 而非写回，确保重试时从 DB 读取最新状态）
     */
    private void markStuckTasksAsFailed(LambdaQueryWrapper<SysTask> wrapper, String errorMsg) {
        try {
            List<SysTask> tasks = sysTaskMapper.selectList(wrapper);
            if (tasks.isEmpty()) {
                return;
            }

            LocalDateTime now = LocalDateTime.now();
            for (SysTask task : tasks) {
                task.setStatus(TaskConstants.STATUS_FAILED);
                task.setErrorMessage(errorMsg);
                task.setCompletedAt(now);
                sysTaskMapper.updateById(task);

                // 清除缓存：用户重试后从 DB 读取最新状态，避免读到旧 FAILED 缓存
                if (StrUtil.isNotBlank(task.getTaskId())) {
                    redisTemplate.delete(TaskConstants.TASK_CACHE_PREFIX + task.getTaskId());
                }
            }
            log.warn("清理僵死任务[{}]: 共{}条", errorMsg, tasks.size());
        } catch (Exception e) {
            log.error("标记僵死任务 FAILED 失败[{}]", errorMsg, e);
        }
    }
}
