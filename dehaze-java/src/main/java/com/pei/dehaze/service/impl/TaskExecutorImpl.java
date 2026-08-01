package com.pei.dehaze.service.impl;

import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.config.WebSocketMessageRelay;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.mq.RabbitMQPublisher;
import com.pei.dehaze.service.TaskExecutor;
import com.pei.dehaze.service.strategy.DefaultProgressCallback;
import com.pei.dehaze.service.strategy.ProgressCallback;
import com.pei.dehaze.service.strategy.TaskCancelledException;
import com.pei.dehaze.service.strategy.TaskResult;
import com.pei.dehaze.service.strategy.TaskStrategy;
import com.pei.dehaze.service.strategy.TaskStrategyFactory;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * 任务执行器实现（策略模式 + MQ 统一路径）
 *
 * <p>发布任务到 RabbitMQ，由 Consumer 调用 {@link #executeExportTask} 执行。
 * MQ 未启用时（测试环境）fallback 到同步执行。
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Slf4j
@Component
public class TaskExecutorImpl implements TaskExecutor {

    private final SysTaskMapper sysTaskMapper;

    private final TaskStrategyFactory strategyFactory;

    private final RedisTemplate<String, Object> redisTemplate;

    private final WebSocketMessageRelay wsMessageRelay;

    /**
     * MQ 发布器（MQ 未启用时为空，fallback 到同步执行）
     */
    private final ObjectProvider<RabbitMQPublisher> publisherProvider;

    public TaskExecutorImpl(SysTaskMapper sysTaskMapper,
                            TaskStrategyFactory strategyFactory,
                            RedisTemplate<String, Object> redisTemplate,
                            WebSocketMessageRelay wsMessageRelay,
                            ObjectProvider<RabbitMQPublisher> publisherProvider) {
        this.sysTaskMapper = sysTaskMapper;
        this.strategyFactory = strategyFactory;
        this.redisTemplate = redisTemplate;
        this.wsMessageRelay = wsMessageRelay;
        this.publisherProvider = publisherProvider;
    }

    @Override
    public void publishExportTask(Long taskId) {
        RabbitMQPublisher publisher = publisherProvider.getIfAvailable();
        if (publisher == null) {
            // MQ 未启用（测试环境），直接同步执行
            log.warn("MQ 未启用，直接同步执行任务: taskId={}", taskId);
            executeExportTask(taskId, null);
            return;
        }

        SysTask sysTask = sysTaskMapper.selectById(taskId);
        if (sysTask == null) {
            log.error("任务不存在，无法发布到 MQ: taskId={}", taskId);
            return;
        }

        String traceId = MDC.get("trace_id");
        Map<String, Object> messageBody = new HashMap<>();
        messageBody.put("db_task_id", taskId);
        messageBody.put("task_id", sysTask.getTaskId());
        messageBody.put("task_type", sysTask.getTaskType());
        String payload = JSONUtil.toJsonStr(messageBody);
        publisher.publish("task.export", payload, traceId);
        log.debug("任务已发布到 MQ: dbTaskId={}, taskId={}, type={}, traceId={}",
                taskId, sysTask.getTaskId(), sysTask.getTaskType(), traceId);
    }

    @Override
    public void executeExportTask(Long taskId, ExportTaskCreateForm form) {
        log.debug("开始执行任务: taskId={}, thread={}", taskId, Thread.currentThread().getName());

        SysTask sysTask = sysTaskMapper.selectById(taskId);
        if (sysTask == null) {
            log.error("任务不存在: taskId={}", taskId);
            return;
        }

        // form 为空时从 DB 任务参数重建（MQ Consumer 调用路径）
        if (form == null) {
            form = JSONUtil.toBean(sysTask.getParams(), ExportTaskCreateForm.class);
        }

        try {
            // 更新任务状态为处理中
            updateTaskStatus(sysTask, TaskConstants.STATUS_PROCESSING, null, null);

            // 获取对应的策略
            TaskStrategy strategy = strategyFactory.getStrategy(form.getType());

            // 创建进度回调（传入 relay 用于 WebSocket 推送 + userId 用于用户定向推送）
            ProgressCallback callback = new DefaultProgressCallback(
                    taskId, sysTask.getTaskId(), sysTask.getCreateBy(),
                    sysTaskMapper, redisTemplate, wsMessageRelay
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
     * <p>优先使用 paramsJson（通用导入导出框架使用），其次回退到 form 整体序列化（数据集导出/下载使用）
     */
    private Map<String, Object> buildParams(ExportTaskCreateForm form) {
        if (form.getParamsJson() != null && !form.getParamsJson().isBlank()) {
            return JSONUtil.toBean(form.getParamsJson(), Map.class);
        }
        String json = JSONUtil.toJsonStr(form);
        return JSONUtil.toBean(json, Map.class);
    }

    /**
     * 更新任务状态
     */
    private void updateTaskStatus(SysTask task, int status, String result, String errorMessage) {
        task.setStatus(status);

        LocalDateTime now = LocalDateTime.now();

        switch (status) {
            case TaskConstants.STATUS_PROCESSING -> task.setStartedAt(now);
            case TaskConstants.STATUS_COMPLETED -> {
                task.setProgress(100);
                task.setResult(toJsonSafe(result));
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

        // WebSocket 推送任务状态变更（通过 Redis Pub/Sub 跨实例投递）
        pushTaskStatusMessage(task, status, result, errorMessage);
    }

    /**
     * 将结果字符串安全转为合法 JSON 值存入 sys_task.result (JSON 列)
     * <ul>
     *   <li>null/空 → {@code "null"}（MySQL JSON 列接受 JSON null）</li>
     *   <li>已是合法 JSON → 原样返回</li>
     *   <li>普通字符串 → JSON 字符串编码（如 {@code "exports/file.zip"} → {@code "\"exports/file.zip\""}）</li>
     * </ul>
     */
    private String toJsonSafe(String result) {
        if (result == null || result.isBlank()) {
            return "null";
        }
        if (JSONUtil.isTypeJSON(result)) {
            return result;
        }
        return JSONUtil.toJsonStr(result);
    }

    /**
     * 通过 WebSocket 推送任务状态变更（对齐 Python 消息格式）
     */
    private void pushTaskStatusMessage(SysTask task, int status, String result, String errorMessage) {
        try {
            Map<String, Object> message = new HashMap<>();
            message.put("type", "task_status");
            message.put("task_id", task.getTaskId());
            message.put("status", status);
            message.put("progress", task.getProgress());
            message.put("result", result);
            message.put("error_message", errorMessage);
            message.put("timestamp", LocalDateTime.now().toString());
            wsMessageRelay.publishToUser(task.getCreateBy(), WebSocketMessageRelay.DEST_TASK, message);
        } catch (Exception e) {
            log.debug("WebSocket 推送失败（不影响任务执行）: {}", e.getMessage());
        }
    }
}
