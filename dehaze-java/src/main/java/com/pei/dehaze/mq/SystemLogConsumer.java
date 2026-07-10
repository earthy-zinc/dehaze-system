package com.pei.dehaze.mq;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * 系统日志消费者
 * <p>
 * 消费 dehaze.logs Topic，处理系统日志
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Component
@ConditionalOnProperty(prefix = "kafka.log-pipeline", name = "enabled", havingValue = "true")
public class SystemLogConsumer extends KafkaLogConsumer {

    public SystemLogConsumer(ObjectMapper objectMapper) {
        super(objectMapper);
    }

    /**
     * 消费系统日志
     * <p>
     * 监听 dehaze.logs Topic
     */
    @KafkaListener(topics = "${kafka.log-pipeline.topic:dehaze.logs}", groupId = "${kafka.consumer.group-id:dehaze-java}")
    public void onLogMessage(ConsumerRecord<String, String> record, Acknowledgment acknowledgment) {
        String topic = record.topic();
        String payload = record.value();

        log.debug("收到日志消息: topic={}, partition={}, offset={}",
                topic, record.partition(), record.offset());

        try {
            Map<String, Object> logEntry = parseLogEntry(payload);
            if (logEntry == null) {
                acknowledge(acknowledgment);
                return;
            }

            String level = extractField(logEntry, "level", String.class);
            String message = extractField(logEntry, "message", String.class);
            String traceId = extractField(logEntry, "traceId", String.class);
            String service = extractField(logEntry, "service", String.class);

            // 处理日志（可根据级别分发到不同处理逻辑）
            processLog(level, message, traceId, service, logEntry);

            acknowledge(acknowledgment);
        } catch (Exception e) {
            handleError(topic, record, e);
            acknowledge(acknowledgment); // 即使失败也确认，避免重复消费
        }
    }

    /**
     * 处理日志
     */
    private void processLog(String level, String message, String traceId,
            String service, Map<String, Object> logEntry) {
        // 根据日志级别处理
        switch (level != null ? level.toUpperCase() : "INFO") {
            case "ERROR":
                log.error("[{}] {} - traceId={}", service, message, traceId);
                break;
            case "WARN":
                log.warn("[{}] {} - traceId={}", service, message, traceId);
                break;
            case "DEBUG":
                log.debug("[{}] {} - traceId={}", service, message, traceId);
                break;
            default:
                log.info("[{}] {} - traceId={}", service, message, traceId);
        }

        // TODO: 可以将日志存储到 Elasticsearch 或其他存储
    }
}
