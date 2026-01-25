package com.pei.dehaze.job;

import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 任务定时清理任务
 * 定期清理过期和长时间未更新的导出任务
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Slf4j
@Component
public class TaskCleanupJob {

    private static final String TASK_CACHE_PREFIX = "export:task:";

    @Resource
    private SysTaskMapper sysTaskMapper;

    @Resource
    private RedisTemplate<String, Object> redisTemplate;

    /**
     * 每天凌晨2点执行清理任务
     * 清理7天前的已完成任务和30天前的所有任务
     */
    @Scheduled(cron = "0 0 2 * * ?")
    public void cleanupExpiredTasks() {
        log.info("开始清理过期导出任务...");

        try {
            // 清理7天前的已完成任务
            LocalDateTime sevenDaysAgo = LocalDateTime.now().minusDays(7);
            List<SysTask> completedTasks = sysTaskMapper.selectList(
                    new LambdaQueryWrapper<SysTask>()
                            .in(SysTask::getStatus, "completed", "failed", "cancelled")
                            .lt(SysTask::getCompletedAt, sevenDaysAgo)
            );

            if (!completedTasks.isEmpty()) {
                int deletedCount = 0;
                for (SysTask task : completedTasks) {
                    // 删除MySQL中的任务记录
                    sysTaskMapper.deleteById(task.getId());

                    // 删除Redis缓存
                    String cacheKey = TASK_CACHE_PREFIX + task.getTaskId();
                    redisTemplate.delete(cacheKey);

                    deletedCount++;
                }
                log.info("清理已完成任务完成: 共清理{}条记录", deletedCount);
            }

            // 清理30天前的所有任务（包括未完成的）
            LocalDateTime thirtyDaysAgo = LocalDateTime.now().minusDays(30);
            List<SysTask> oldTasks = sysTaskMapper.selectList(
                    new LambdaQueryWrapper<SysTask>()
                            .lt(SysTask::getCreatedAt, thirtyDaysAgo)
            );

            if (!oldTasks.isEmpty()) {
                int deletedCount = 0;
                for (SysTask task : oldTasks) {
                    // 删除MySQL中的任务记录
                    sysTaskMapper.deleteById(task.getId());

                    // 删除Redis缓存
                    if (StrUtil.isNotBlank(task.getTaskId())) {
                        String cacheKey = TASK_CACHE_PREFIX + task.getTaskId();
                        redisTemplate.delete(cacheKey);
                    }

                    deletedCount++;
                }
                log.info("清理旧任务完成: 共清理{}条记录", deletedCount);
            }

            log.info("清理过期导出任务完成");

        } catch (Exception e) {
            log.error("清理过期任务失败", e);
        }
    }

    /**
     * 每小时执行一次，清理异常任务
     * 清理超过24小时还在pending或processing状态的任务
     */
    @Scheduled(cron = "0 0 * * * ?")
    public void cleanupStuckTasks() {
        log.info("开始清理异常状态的任务...");

        try {
            LocalDateTime oneDayAgo = LocalDateTime.now().minusDays(1);

            // 查找超过24小时还在pending或processing状态的任务
            List<SysTask> stuckTasks = sysTaskMapper.selectList(
                    new LambdaQueryWrapper<SysTask>()
                            .in(SysTask::getStatus, "pending", "processing")
                            .lt(SysTask::getCreatedAt, oneDayAgo)
            );

            if (!stuckTasks.isEmpty()) {
                int updatedCount = 0;
                for (SysTask task : stuckTasks) {
                    // 将任务标记为失败
                    task.setStatus("failed");
                    task.setErrorMessage("任务执行超时");
                    task.setCompletedAt(LocalDateTime.now());
                    sysTaskMapper.updateById(task);

                    // 更新Redis缓存
                    if (StrUtil.isNotBlank(task.getTaskId())) {
                        String cacheKey = TASK_CACHE_PREFIX + task.getTaskId();
                        redisTemplate.opsForValue().set(cacheKey, task);
                    }

                    updatedCount++;
                }
                log.warn("清理异常任务完成: 共清理{}条记录", updatedCount);
            } else {
                log.info("没有发现异常状态的任务");
            }

        } catch (Exception e) {
            log.error("清理异常任务失败", e);
        }
    }
}
