package com.pei.dehaze.mq;

import com.pei.dehaze.filter.TraceIdFilter;
import com.pei.dehaze.security.util.SystemSecurityContext;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.amqp.core.Message;

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
     * @param queueName 队列名称（用于日志）
     * @param handler   业务处理逻辑
     */
    protected void processMessage(Message message, String queueName, MessageHandler handler) {
        SystemSecurityContext.setSystemContext();
        String traceId = extractTraceId(message);
        if (traceId != null) {
            MDC.put(TraceIdFilter.MDC_TRACE_ID, traceId);
        }
        try {
            String body = extractBody(message);
            handler.handle(body, traceId);
        } catch (Exception e) {
            handleError(queueName, traceId, e);
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

    /**
     * 处理消息异常（记录日志，消息将被确认）
     */
    protected void handleError(String queueName, String traceId, Exception e) {
        log.error("RabbitMQ 消息处理失败: queue={}, traceId={}", queueName, traceId, e);
    }
}
