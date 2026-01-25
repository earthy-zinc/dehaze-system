package com.pei.dehaze.service.impl;

import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.service.TaskExecutor;
import com.pei.dehaze.service.strategy.DefaultProgressCallback;
import com.pei.dehaze.service.strategy.ProgressCallback;
import com.pei.dehaze.service.strategy.TaskCancelledException;
import com.pei.dehaze.service.strategy.TaskResult;
import com.pei.dehaze.service.strategy.TaskStrategy;
import com.pei.dehaze.service.strategy.TaskStrategyFactory;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * 任务执行器实现（策略模式重构版）
 * 仅负责任务调度，业务逻辑委托给具体策略
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Slf4j
@Component
public class TaskExecutorImpl implements TaskExecutor {

    @Resource
    private SysTaskMapper sysTaskMapper;

    @Resource
    private TaskStrategyFactory strategyFactory;

    @Resource
    private RedisTemplate<String, Object> redisTemplate;

    @Override
    @Async("datasetTaskExecutor")
    public void submitExportTask(Long taskId, ExportTaskCreateForm form) {
        log.info("开始执行任务: taskId={}, type={}, thread={}",
                taskId, form.getType(), Thread.currentThread().getName());

        SysTask sysTask = sysTaskMapper.selectById(taskId);
        if (sysTask == null) {
            log.error("任务不存在: taskId={}", taskId);
            return;
        }

        try {
            // 更新任务状态为处理中
            updateTaskStatus(sysTask, TaskConstants.STATUS_PROCESSING, null, null);
            
            // 获取对应的策略
            TaskStrategy strategy = strategyFactory.getStrategy(form.getType());

            // 创建进度回调
            ProgressCallback callback = new DefaultProgressCallback(
                    taskId, sysTask.getTaskId(), sysTaskMapper, redisTemplate
            );

            // 解析参数为 Map
            Map<String, Object> params = buildParams(form);

            // 验证参数
            strategy.validateParams(params);

            // 执行任务
            TaskResult result = strategy.execute(sysTask, params, callback);

            // 处理执行结果
            if (result.isSuccess()) {
                updateTaskStatus(sysTask, TaskConstants.STATUS_COMPLETED, result.getData(), null);
                log.info("任务执行成功: taskId={}, result={}", taskId, result.getData());
            } else {
                updateTaskStatus(sysTask, TaskConstants.STATUS_FAILED, null, result.getErrorMessage());
                log.error("任务执行失败: taskId={}, error={}", taskId, result.getErrorMessage());
            }

        } catch (TaskCancelledException e) {
            log.warn("任务被取消: taskId={}", taskId);
            updateTaskStatus(sysTask, TaskConstants.STATUS_CANCELLED, null, null);
        } catch (Exception e) {
            log.error("任务执行异常: taskId={}", taskId, e);
            updateTaskStatus(sysTask, TaskConstants.STATUS_FAILED, null, e.getMessage());
        }
    }

    /**
     * 构建参数Map
     */
    private Map<String, Object> buildParams(ExportTaskCreateForm form) {
        String json = JSONUtil.toJsonStr(form);
        return JSONUtil.toBean(json, Map.class);
    }

    /**
     * 更新任务状态
     */
    private void updateTaskStatus(SysTask task, String status, String result, String errorMessage) {
        task.setStatus(status);
        
        LocalDateTime now = LocalDateTime.now();
        
        switch (status) {
            case TaskConstants.STATUS_PROCESSING -> task.setStartedAt(now);
            case TaskConstants.STATUS_COMPLETED -> {
                task.setProgress(100);
                task.setResult(result);
                task.setCompletedAt(now);
                task.setExpiresAt(now.plusDays(7)); // 结果保留7天
            }
            case TaskConstants.STATUS_FAILED -> {
                task.setErrorMessage(errorMessage);
                task.setCompletedAt(now);
            }
            case TaskConstants.STATUS_CANCELLED -> task.setCompletedAt(now);
        }

        sysTaskMapper.updateById(task);

        // 更新Redis缓存
        String cacheKey = TaskConstants.TASK_CACHE_PREFIX + task.getTaskId();
        redisTemplate.opsForValue().set(cacheKey, task, TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);
    }
}
