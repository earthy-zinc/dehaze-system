package com.pei.dehaze.service.strategy;

import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.config.WebSocketMessageRelay;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * 默认进度回调实现
 * 支持进度节流、取消检测和 WebSocket 进度推送
 * <p>
 * 对齐 Python 端策略：进度更新时同步刷新 Redis 缓存（避免 getTaskStatus 读到过期进度），
 * 通过 Redis Pub/Sub 推送到任务创建者（跨实例投递）。
 */
@Slf4j
public class DefaultProgressCallback implements ProgressCallback {

    private static final long UPDATE_INTERVAL_MS = 2000; // 进度更新间隔2秒
    private static final int PROGRESS_COMPLETE = 100;
    private static final int PROGRESS_UPDATE_THRESHOLD = 5;

    private final Long taskId;
    private final String taskIdStr;
    private final Long createBy;
    private final SysTaskMapper taskMapper;
    private final RedisTemplate<String, Object> redisTemplate;
    private final WebSocketMessageRelay wsMessageRelay;
    private final String cancelKey;
    private final String cacheKey;

    private long lastUpdateTime = 0;
    private int lastProgress = -1;

    public DefaultProgressCallback(Long taskId, String taskIdStr, Long createBy,
                                   SysTaskMapper taskMapper,
                                   RedisTemplate<String, Object> redisTemplate,
                                   WebSocketMessageRelay wsMessageRelay) {
        this.taskId = taskId;
        this.taskIdStr = taskIdStr;
        this.createBy = createBy;
        this.taskMapper = taskMapper;
        this.redisTemplate = redisTemplate;
        this.wsMessageRelay = wsMessageRelay;
        this.cancelKey = TaskConstants.TASK_CANCEL_PREFIX + taskIdStr;
        this.cacheKey = TaskConstants.TASK_CACHE_PREFIX + taskIdStr;
    }

    @Override
    public void updateProgress(int current, int total, String message) {
        // 先检查取消状态
        checkCancelled();

        int progress = total > 0 ? (current * PROGRESS_COMPLETE / total) : PROGRESS_COMPLETE;
        long now = System.currentTimeMillis();

        // 节流：进度变化>=5% 或 时间间隔>=2秒 或 已完成时才更新
        boolean shouldUpdate = (progress - lastProgress >= PROGRESS_UPDATE_THRESHOLD)
                || (now - lastUpdateTime >= UPDATE_INTERVAL_MS)
                || (progress >= PROGRESS_COMPLETE);

        if (!shouldUpdate) {
            return;
        }

        lastUpdateTime = now;
        lastProgress = progress;

        // 更新数据库
        SysTask updateTask = new SysTask();
        updateTask.setId(taskId);
        updateTask.setProgress(progress);
        updateTask.setProcessedFiles(current);
        updateTask.setTotalFiles(total);
        taskMapper.updateById(updateTask);

        // 同步更新 Redis 缓存中的进度字段（修复：原实现只更新 DB 不更新缓存导致 getTaskStatus 读到过期进度）
        try {
            SysTask cachedTask = (SysTask) redisTemplate.opsForValue().get(cacheKey);
            if (cachedTask != null) {
                cachedTask.setProgress(progress);
                cachedTask.setProcessedFiles(current);
                cachedTask.setTotalFiles(total);
                redisTemplate.opsForValue().set(cacheKey, cachedTask,
                        TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);
            }
        } catch (Exception e) {
            log.warn("Redis 缓存进度刷新失败（不影响任务执行）", e);
        }

        // WebSocket 推送任务进度（对齐 Python 消息格式）
        pushProgressMessage(progress, current, total);

        log.debug("任务进度更新: taskId={}, progress={}/{} ({}%)",
                taskIdStr, current, total, progress);
    }

    /**
     * 通过 WebSocket 推送任务进度（对齐 Python 消息格式）
     */
    private void pushProgressMessage(int progress, int current, int total) {
        if (wsMessageRelay == null) {
            return;
        }
        try {
            Map<String, Object> msg = new HashMap<>();
            msg.put("type", "task_progress");
            msg.put("task_id", taskIdStr);
            msg.put("progress", progress);
            msg.put("status", TaskConstants.STATUS_PROCESSING);
            msg.put("processed_files", current);
            msg.put("total_files", total);
            msg.put("timestamp", LocalDateTime.now().toString());
            wsMessageRelay.publishToUser(createBy, msg);
        } catch (Exception e) {
            log.warn("WebSocket 进度推送失败（不影响任务执行）", e);
        }
    }

    @Override
    public boolean isCancelled() {
        Boolean cancelled = (Boolean) redisTemplate.opsForValue().get(cancelKey);
        return Boolean.TRUE.equals(cancelled);
    }
}
