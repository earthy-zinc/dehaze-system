package com.pei.dehaze.mq;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

/**
 * 缩略图任务消费者
 * <p>
 * 消费缩略图生成队列消息，执行异步缩略图生成操作
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "rabbitmq", name = "enabled", havingValue = "true")
public class ThumbnailTaskConsumer extends RabbitMQConsumer {

    /**
     * 消费缩略图任务队列
     * <p>
     * 监听 task.thumbnail 队列，处理缩略图生成任务
     */
    @RabbitListener(queues = "task.thumbnail")
    public void onThumbnailMessage(Message message) {
        String body = extractBody(message);
        String traceId = extractTraceId(message);

        log.info("收到缩略图生成任务: traceId={}", traceId);

        try {
            // TODO: 调用缩略图服务处理任务
            // thumbnailService.processThumbnail(body, traceId);
            log.debug("缩略图生成任务处理完成: traceId={}", traceId);
        } catch (Exception e) {
            handleError("task.thumbnail", traceId, e);
        }
    }
}
