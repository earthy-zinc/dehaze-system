package com.pei.dehaze.service.strategy;

import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;

/**
 * 默认进度回调实现
 * 支持进度节流和取消检测
 */
@Slf4j
public class DefaultProgressCallback implements ProgressCallback {

    private static final long UPDATE_INTERVAL_MS = 2000; // 进度更新间隔2秒

    private final Long taskId;
    private final String taskIdStr;
    private final SysTaskMapper taskMapper;
    private final RedisTemplate<String, Object> redisTemplate;
    private final String cancelKey;

    private long lastUpdateTime = 0;
    private int lastProgress = -1;

    public DefaultProgressCallback(Long taskId, String taskIdStr,
                                   SysTaskMapper taskMapper,
                                   RedisTemplate<String, Object> redisTemplate) {
        this.taskId = taskId;
        this.taskIdStr = taskIdStr;
        this.taskMapper = taskMapper;
        this.redisTemplate = redisTemplate;
        this.cancelKey = TaskConstants.TASK_CANCEL_PREFIX + taskIdStr;
    }

    @Override
    public void updateProgress(int current, int total, String message) {
        // 先检查取消状态
        checkCancelled();

        int progress = total > 0 ? (current * 100 / total) : 100;
        long now = System.currentTimeMillis();

        // 节流：进度变化>=5% 或 时间间隔>=2秒 或 已完成时才更新
        boolean shouldUpdate = (progress - lastProgress >= 5)
                || (now - lastUpdateTime >= UPDATE_INTERVAL_MS)
                || (progress >= 100);

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

        log.debug("任务进度更新: taskId={}, progress={}/{} ({}%), message={}",
                taskIdStr, current, total, progress, message);
    }

    @Override
    public boolean isCancelled() {
        Boolean cancelled = (Boolean) redisTemplate.opsForValue().get(cancelKey);
        return Boolean.TRUE.equals(cancelled);
    }
}
