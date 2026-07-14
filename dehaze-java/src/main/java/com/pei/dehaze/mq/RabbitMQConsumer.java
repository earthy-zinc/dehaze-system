package com.pei.dehaze.mq;

import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * RabbitMQ 消息消费者基类
 * <p>
 * 提供消息消费的基础设施，支持 TraceID 传递
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
public abstract class RabbitMQConsumer {

    /**
     * 从消息头提取 TraceID
     */
    protected String extractTraceId(Message message) {
        Map<String, Object> headers = message.getMessageProperties().getHeaders();
        if (headers == null) {
            return null;
        }
        Object traceId = headers.get("X-Trace-Id");
        if (traceId != null) {
            return traceId.toString();
        }
        return null;
    }

    /**
     * 从消息体提取字符串
     */
    protected String extractBody(Message message) {
        return new String(message.getBody(), StandardCharsets.UTF_8);
    }

    /**
     * 处理消息异常（记录日志，消息将被确认）
     */
    protected void handleError(String queueName, String traceId, Exception e) {
        log.error("RabbitMQ 消息处理失败: queue={}, traceId={}", queueName, traceId, e);
    }
}
