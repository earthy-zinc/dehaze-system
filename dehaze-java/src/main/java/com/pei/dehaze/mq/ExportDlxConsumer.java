package com.pei.dehaze.mq;

import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.config.WebSocketMessageRelay;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.security.util.SystemSecurityContext;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import com.pei.dehaze.filter.TraceIdFilter;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * 导出任务死信队列消费者
 * <p>
 * 消费 task.export.dlx 队列消息，将重试耗尽或过期的任务标记为 FAILED。
 * 对齐 Python handle_dlq_message 行为。
 *
 * @author earthyzinc
 * @since 2026-07-14
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "rabbitmq", name = "enabled", havingValue = "true")
public class ExportDlxConsumer extends RabbitMQConsumer {

    private static final Set<String> TERMINAL_STATUSES = Set.of(
            TaskConstants.STATUS_COMPLETED,
            TaskConstants.STATUS_FAILED,
            TaskConstants.STATUS_CANCELLED
    );

    private final SysTaskMapper sysTaskMapper;
    private final RedisTemplate<String, Object> redisTemplate;
    private final WebSocketMessageRelay wsMessageRelay;

    @RabbitListener(queues = "task.export.dlx")
    public void onDlxMessage(Message message) {
        SystemSecurityContext.setSystemContext();
        String traceId = extractTraceId(message);
        if (traceId != null) {
            MDC.put(TraceIdFilter.MDC_TRACE_ID, traceId);
        }
        try {
            String body = extractBody(message);
            Long taskId;
            try {
                taskId = Long.parseLong(body.trim());
            } catch (NumberFormatException e) {
                log.error("[DLQ] 死信消息体无法解析为 taskId: body={}", body);
                return;
            }

            log.warn("[DLQ] 收到死信消息: taskId={}, traceId={}", taskId, traceId);

            SysTask sysTask = sysTaskMapper.selectById(taskId);
            if (sysTask == null) {
                log.warn("[DLQ] 任务不存在，无法标记失败: taskId={}", taskId);
                return;
            }

            // 仅在非终态时更新
            if (TERMINAL_STATUSES.contains(sysTask.getStatus())) {
                log.info("[DLQ] 任务已为终态，跳过: taskId={}, status={}", taskId, sysTask.getStatus());
                return;
            }

            // 标记为 FAILED
            sysTask.setStatus(TaskConstants.STATUS_FAILED);
            sysTask.setErrorMessage("消息重试耗尽进入死信队列");
            sysTask.setCompletedAt(LocalDateTime.now());
            sysTaskMapper.updateById(sysTask);

            // 更新缓存
            String cacheKey = TaskConstants.TASK_CACHE_PREFIX + sysTask.getTaskId();
            redisTemplate.opsForValue().set(cacheKey, sysTask, TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);

            // WebSocket 推送失败通知
            pushFailedMessage(sysTask);

            log.warn("[DLQ] 死信消息处理完成，任务已标记失败: taskId={}", taskId);
        } catch (Exception e) {
            handleError("task.export.dlx", traceId, e);
        } finally {
            MDC.remove(TraceIdFilter.MDC_TRACE_ID);
            SystemSecurityContext.clearContext();
        }
    }

    private void pushFailedMessage(SysTask sysTask) {
        try {
            Map<String, Object> message = new HashMap<>();
            message.put("type", "task_status");
            message.put("task_id", sysTask.getTaskId());
            message.put("status", TaskConstants.STATUS_FAILED);
            message.put("progress", sysTask.getProgress());
            message.put("result", null);
            message.put("error_message", sysTask.getErrorMessage());
            message.put("timestamp", LocalDateTime.now().toString());
            wsMessageRelay.publishToUser(sysTask.getCreateBy(), message);
        } catch (Exception e) {
            log.debug("[DLQ] WebSocket 推送失败: {}", e.getMessage());
        }
    }
}
