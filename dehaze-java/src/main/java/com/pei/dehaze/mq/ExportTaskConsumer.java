package com.pei.dehaze.mq;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

/**
 * 导出任务消费者
 * <p>
 * 消费导出任务队列消息，执行异步导出操作
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "rabbitmq", name = "enabled", havingValue = "true")
public class ExportTaskConsumer extends RabbitMQConsumer {

    /**
     * 消费导出任务队列
     * <p>
     * 监听 task.export 队列，处理导出任务
     */
    @RabbitListener(queues = "task.export")
    public void onExportMessage(Message message) {
        String body = extractBody(message);
        String traceId = extractTraceId(message);

        log.info("收到导出任务: traceId={}", traceId);

        try {
            // TODO: 调用导出服务处理任务
            // exportService.processExport(body, traceId);
            log.debug("导出任务处理完成: traceId={}", traceId);
        } catch (Exception e) {
            handleError("task.export", traceId, e);
        }
    }
}
