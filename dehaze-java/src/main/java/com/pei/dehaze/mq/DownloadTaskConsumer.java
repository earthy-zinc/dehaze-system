package com.pei.dehaze.mq;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

/**
 * 下载任务消费者
 * <p>
 * 消费批量下载队列消息，执行异步下载操作
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "rabbitmq", name = "enabled", havingValue = "true")
public class DownloadTaskConsumer extends RabbitMQConsumer {

    /**
     * 消费下载任务队列
     * <p>
     * 监听 task.download 队列，处理批量下载任务
     */
    @RabbitListener(queues = "task.download")
    public void onDownloadMessage(Message message) {
        String body = extractBody(message);
        String traceId = extractTraceId(message);

        log.info("收到下载任务: traceId={}", traceId);

        try {
            // TODO: 调用下载服务处理任务
            // downloadService.processDownload(body, traceId);
            log.debug("下载任务处理完成: traceId={}", traceId);
        } catch (Exception e) {
            handleError("task.download", traceId, e);
        }
    }
}
