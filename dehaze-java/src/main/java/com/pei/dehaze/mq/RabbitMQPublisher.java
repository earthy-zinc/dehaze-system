package com.pei.dehaze.mq;

import com.pei.dehaze.config.property.RabbitMQProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.core.MessageBuilder;
import org.springframework.amqp.core.MessageProperties;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

/**
 * RabbitMQ 消息发布器
 * <p>
 * 用于异步任务分发（导出、批量操作等）
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "rabbitmq", name = "enabled", havingValue = "true")
public class RabbitMQPublisher {

    private final RabbitTemplate rabbitTemplate;
    private final RabbitMQProperties properties;

    /**
     * 发布消息到指定队列
     *
     * @param queueName 队列名称 (如 export, download, thumbnail)
     * @param payload   消息内容 (JSON 字符串)
     */
    public void publish(String queueName, String payload) {
        publish(queueName, payload, null);
    }

    /**
     * 发布消息到指定队列（带 TraceID）
     *
     * @param queueName 队列名称
     * @param payload   消息内容
     * @param traceId   追踪 ID
     */
    public void publish(String queueName, String payload, String traceId) {
        String routingKey = resolveRoutingKey(queueName);
        String exchange = properties.getExchange().getName();

        Message message = MessageBuilder
                .withBody(payload.getBytes())
                .setContentType(MessageProperties.CONTENT_TYPE_JSON)
                .setDeliveryMode(MessageProperties.DeliveryMode.PERSISTENT)
                .build();

        // 设置 TraceID 到消息头
        if (traceId != null && !traceId.isEmpty()) {
            message.getMessageProperties().setHeader("X-Trace-ID", traceId);
        }

        rabbitTemplate.send(exchange, routingKey, message);
        log.debug("RabbitMQ 消息已发布: exchange={}, routingKey={}, traceId={}",
                exchange, routingKey, traceId);
    }

    /**
     * 发布对象消息（自动序列化为 JSON）
     *
     * @param queueName 队列名称
     * @param object    消息对象
     */
    public void publishObject(String queueName, Object object) {
        String routingKey = resolveRoutingKey(queueName);
        String exchange = properties.getExchange().getName();

        rabbitTemplate.convertAndSend(exchange, routingKey, object, message -> {
            message.getMessageProperties().setDeliveryMode(MessageProperties.DeliveryMode.PERSISTENT);
            return message;
        });

        log.debug("RabbitMQ 对象消息已发布: exchange={}, routingKey={}", exchange, routingKey);
    }

    /**
     * 发布对象消息（带 TraceID）
     *
     * @param queueName 队列名称
     * @param object    消息对象
     * @param traceId   追踪 ID
     */
    public void publishObject(String queueName, Object object, String traceId) {
        String routingKey = resolveRoutingKey(queueName);
        String exchange = properties.getExchange().getName();

        rabbitTemplate.convertAndSend(exchange, routingKey, object, message -> {
            message.getMessageProperties().setDeliveryMode(MessageProperties.DeliveryMode.PERSISTENT);
            if (traceId != null && !traceId.isEmpty()) {
                message.getMessageProperties().setHeader("X-Trace-ID", traceId);
            }
            return message;
        });

        log.debug("RabbitMQ 对象消息已发布: exchange={}, routingKey={}, traceId={}",
                exchange, routingKey, traceId);
    }

    /**
     * 解析路由键
     */
    private String resolveRoutingKey(String queueName) {
        String prefix = properties.getExchange().getRoutingKeyPrefix();
        return prefix + queueName;
    }
}
