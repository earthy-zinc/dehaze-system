package com.pei.dehaze.mq;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.config.property.KafkaProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.clients.producer.RecordMetadata;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Kafka 日志生产者
 * <p>
 * 用于日志收集与流处理
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "kafka", name = "enabled", havingValue = "true")
public class KafkaLogProducer {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final KafkaProperties properties;
    private final ObjectMapper objectMapper;

    /**
     * 发送日志消息
     *
     * @param level   日志级别
     * @param message 日志消息
     * @return CompletableFuture
     */
    public CompletableFuture<SendResult<String, String>> sendLog(String level, String message) {
        return sendLog(level, message, null, null);
    }

    /**
     * 发送日志消息（带上下文）
     *
     * @param level   日志级别
     * @param message 日志消息
     * @param context 上下文信息
     * @param traceId 追踪 ID
     * @return CompletableFuture
     */
    public CompletableFuture<SendResult<String, String>> sendLog(
            String level, String message, Map<String, Object> context, String traceId) {

        Map<String, Object> logEntry = new HashMap<>();
        logEntry.put("timestamp", Instant.now().toString());
        logEntry.put("level", level);
        logEntry.put("message", message);
        logEntry.put("service", "dehaze-java");

        if (traceId != null) {
            logEntry.put("traceId", traceId);
        }

        if (context != null) {
            logEntry.put("context", context);
        }

        String topic = properties.getLogPipeline().getTopic();

        try {
            String payload = objectMapper.writeValueAsString(logEntry);
            return kafkaTemplate.send(topic, traceId, payload)
                    .whenComplete((result, ex) -> {
                        if (ex == null) {
                            RecordMetadata metadata = result.getRecordMetadata();
                            log.debug("日志已发送到 Kafka: topic={}, partition={}, offset={}",
                                    metadata.topic(), metadata.partition(), metadata.offset());
                        } else {
                            log.error("日志发送到 Kafka 失败: {}", ex.getMessage());
                        }
                    });
        } catch (JsonProcessingException e) {
            log.error("日志序列化失败: {}", e.getMessage());
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * 发送审计日志
     *
     * @param userId     用户 ID
     * @param action     操作动作
     * @param resource   资源类型
     * @param resourceId 资源 ID
     * @param details    详细信息
     * @return CompletableFuture
     */
    public CompletableFuture<SendResult<String, String>> sendAuditLog(
            Long userId, String action, String resource, String resourceId, Map<String, Object> details) {

        Map<String, Object> auditEntry = new HashMap<>();
        auditEntry.put("timestamp", Instant.now().toString());
        auditEntry.put("userId", userId);
        auditEntry.put("action", action);
        auditEntry.put("resource", resource);
        auditEntry.put("resourceId", resourceId);
        auditEntry.put("service", "dehaze-java");

        if (details != null) {
            auditEntry.put("details", details);
        }

        String topic = "dehaze.audit";

        try {
            String payload = objectMapper.writeValueAsString(auditEntry);
            return kafkaTemplate.send(topic, String.valueOf(userId), payload);
        } catch (JsonProcessingException e) {
            log.error("审计日志序列化失败: {}", e.getMessage());
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * 发送原始消息
     *
     * @param topic   Topic 名称
     * @param key     消息键
     * @param payload 消息内容
     * @return CompletableFuture
     */
    public CompletableFuture<SendResult<String, String>> send(String topic, String key, String payload) {
        return kafkaTemplate.send(topic, key, payload);
    }

    /**
     * 发送对象消息（自动序列化）
     *
     * @param topic  Topic 名称
     * @param key    消息键
     * @param object 消息对象
     * @return CompletableFuture
     */
    public CompletableFuture<SendResult<String, String>> sendObject(String topic, String key, Object object) {
        try {
            String payload = objectMapper.writeValueAsString(object);
            return kafkaTemplate.send(topic, key, payload);
        } catch (JsonProcessingException e) {
            log.error("消息序列化失败: {}", e.getMessage());
            return CompletableFuture.failedFuture(e);
        }
    }
}
