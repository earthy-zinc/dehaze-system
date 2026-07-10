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
 * 审计日志消费者
 * <p>
 * 消费 dehaze.audit Topic，处理审计日志
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Component
@ConditionalOnProperty(prefix = "kafka", name = "enabled", havingValue = "true")
public class AuditLogConsumer extends KafkaLogConsumer {

    public AuditLogConsumer(ObjectMapper objectMapper) {
        super(objectMapper);
    }

    /**
     * 消费审计日志
     * <p>
     * 监听 dehaze.audit Topic
     */
    @KafkaListener(topics = "dehaze.audit", groupId = "${kafka.consumer.group-id:dehaze-java}-audit")
    public void onAuditMessage(ConsumerRecord<String, String> record, Acknowledgment acknowledgment) {
        String topic = record.topic();
        String payload = record.value();

        log.debug("收到审计日志: topic={}, partition={}, offset={}",
                topic, record.partition(), record.offset());

        try {
            Map<String, Object> auditEntry = parseLogEntry(payload);
            if (auditEntry == null) {
                acknowledge(acknowledgment);
                return;
            }

            Long userId = extractField(auditEntry, "userId", Number.class) != null
                    ? extractField(auditEntry, "userId", Number.class).longValue()
                    : null;
            String action = extractField(auditEntry, "action", String.class);
            String resource = extractField(auditEntry, "resource", String.class);
            String resourceId = extractField(auditEntry, "resourceId", String.class);
            String service = extractField(auditEntry, "service", String.class);

            // 处理审计日志
            processAuditLog(userId, action, resource, resourceId, service, auditEntry);

            acknowledge(acknowledgment);
        } catch (Exception e) {
            handleError(topic, record, e);
            acknowledge(acknowledgment);
        }
    }

    /**
     * 处理审计日志
     */
    private void processAuditLog(Long userId, String action, String resource,
            String resourceId, String service, Map<String, Object> auditEntry) {
        log.info("审计日志: userId={}, action={}, resource={}, resourceId={}, service={}",
                userId, action, resource, resourceId, service);

        // TODO: 可以将审计日志存储到数据库或 Elasticsearch
        // auditLogService.save(auditEntry);
    }
}
