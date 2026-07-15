package com.pei.dehaze.mq;

import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.config.WebSocketMessageRelay;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
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

    private static final String QUEUE_EXPORT_DLX = "task.export.dlx";

    private final SysTaskMapper sysTaskMapper;
    private final RedisTemplate<String, Object> redisTemplate;
    private final WebSocketMessageRelay wsMessageRelay;

    @RabbitListener(queues = QUEUE_EXPORT_DLX)
    public void onDlxMessage(Message message) {
        processMessage(message, QUEUE_EXPORT_DLX, this::handleDlx);
    }

    private void handleDlx(String body, String traceId) throws Exception {
        Long taskId;
        try {
            taskId = Long.parseLong(body.trim());
        } catch (NumberFormatException e) {
            throw new BusinessException("[DLQ] 死信消息体无法解析为 taskId: body=" + body, e);
        }

        log.warn("[DLQ] 收到死信消息: taskId={}, traceId={}", taskId, traceId);

        SysTask sysTask = sysTaskMapper.selectById(taskId);
        if (sysTask == null) {
            log.warn("[DLQ] 任务不存在，无法标记失败: taskId={}", taskId);
            return;
        }

        // 仅在非终态时更新
        if (TaskConstants.TERMINAL_STATUSES.contains(sysTask.getStatus())) {
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
            log.warn("[DLQ] WebSocket 推送失败: taskId={}", sysTask.getTaskId(), e);
        }
    }
}
