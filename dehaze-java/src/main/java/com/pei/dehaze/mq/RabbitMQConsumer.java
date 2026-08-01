package com.pei.dehaze.mq;

import com.pei.dehaze.config.property.RabbitMQProperties;
import com.pei.dehaze.filter.TraceIdFilter;
import com.pei.dehaze.security.util.SystemSecurityContext;
import com.rabbitmq.client.Channel;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.core.MessageProperties;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * RabbitMQ 消息消费者基类
 * <p>
 * 提供消息消费的基础设施，统一处理系统上下文、TraceID 传播与异常日志。
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
public abstract class RabbitMQConsumer {

    @Autowired
    protected RabbitTemplate rabbitTemplate;

    @Autowired
    protected RabbitMQProperties rabbitMQProperties;

    private static final int MAX_RETRY = 3;

    /**
     * 消息处理器：由子类实现具体业务逻辑
     */
    @FunctionalInterface
    protected interface MessageHandler {
        void handle(String body, String traceId) throws Exception;
    }

    /**
     * 消息消费公共流程：设置系统上下文、传播 TraceId 到 MDC、统一异常处理。
     * <p>
     * 子类在 {@code @RabbitListener} 入口直接委托给本方法，避免重复模板代码。
     *
     * @param message   原始消息
     * @param channel   RabbitMQ 通道（用于手动 ack/nack）
     * @param queueName 队列名称（用于日志）
     * @param handler   业务处理逻辑
     */
    protected void processMessage(Message message, Channel channel, String queueName, MessageHandler handler) {
        SystemSecurityContext.setSystemContext();
        String traceId = extractTraceId(message);
        if (traceId != null) {
            MDC.put(TraceIdFilter.MDC_TRACE_ID, traceId);
        }
        long deliveryTag = message.getMessageProperties().getDeliveryTag();
        try {
            String body = extractBody(message);
            handler.handle(body, traceId);
            channel.basicAck(deliveryTag, false);
        } catch (Exception e) {
            handleRetryOrDlx(message, channel, queueName, traceId, deliveryTag, e);
        } finally {
            MDC.remove(TraceIdFilter.MDC_TRACE_ID);
            SystemSecurityContext.clearContext();
        }
    }

    protected void processDlxMessage(Message message, Channel channel, String queueName, MessageHandler handler) {
        SystemSecurityContext.setSystemContext();
        String traceId = extractTraceId(message);
        if (traceId != null) {
            MDC.put(TraceIdFilter.MDC_TRACE_ID, traceId);
        }
        long deliveryTag = message.getMessageProperties().getDeliveryTag();
        try {
            String body = extractBody(message);
            handler.handle(body, traceId);
            channel.basicAck(deliveryTag, false);
        } catch (Exception e) {
            log.error("RabbitMQ DLQ 消息处理失败: queue={}, traceId={}", queueName, traceId, e);
            try {
                channel.basicNack(deliveryTag, false, false);
            } catch (IOException ex) {
                log.error("DLQ nack 失败: queue={}, traceId={}", queueName, traceId, ex);
            }
        } finally {
            MDC.remove(TraceIdFilter.MDC_TRACE_ID);
            SystemSecurityContext.clearContext();
        }
    }

    /**
     * 从消息头提取 TraceID
     */
    protected String extractTraceId(Message message) {
        Map<String, Object> headers = message.getMessageProperties().getHeaders();
        Object traceId = headers.get(TraceIdFilter.TRACE_ID_HEADER);
        return traceId != null ? traceId.toString() : null;
    }

    /**
     * 从消息体提取字符串
     */
    protected String extractBody(Message message) {
        return new String(message.getBody(), StandardCharsets.UTF_8);
    }

    private void handleRetryOrDlx(Message message, Channel channel, String queueName,
                                  String traceId, long deliveryTag, Exception e) {
        log.error("RabbitMQ 消息处理失败: queue={}, traceId={}", queueName, traceId, e);
        try {
            MessageProperties props = message.getMessageProperties();
            Map<String, Object> headers = props.getHeaders();
            Integer retryCount = (Integer) headers.get("x-retry-count");
            if (retryCount == null) {
                retryCount = 0;
            }
            if (retryCount < MAX_RETRY) {
                String retryRoutingKey = resolveRoutingKey(queueName + ".retry." + retryCount);
                final int nextRetryCount = retryCount + 1;
                rabbitTemplate.convertAndSend(
                        rabbitMQProperties.getExchange().getName(),
                        retryRoutingKey,
                        message,
                        m -> {
                            m.getMessageProperties().getHeaders().put("x-retry-count", nextRetryCount);
                            return m;
                        }
                );
                channel.basicAck(deliveryTag, false);
                log.debug("消息已投递到重试队列: queue={}.retry.{}, retryCount={}, traceId={}",
                        queueName, retryCount, retryCount + 1, traceId);
            } else {
                channel.basicNack(deliveryTag, false, false);
                log.warn("消息重试耗尽，已投递到死信队列: queue={}.dlx, totalRetries={}, traceId={}",
                        queueName, retryCount, traceId);
            }
        } catch (IOException ex) {
            log.error("重试/死信处理失败: queue={}, traceId={}", queueName, traceId, ex);
            try {
                channel.basicNack(deliveryTag, false, false);
            } catch (IOException ignore) {
            }
        }
    }

    private String resolveRoutingKey(String queueName) {
        return queueName;
    }
}
