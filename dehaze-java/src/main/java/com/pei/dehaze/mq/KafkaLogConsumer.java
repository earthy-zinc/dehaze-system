package com.pei.dehaze.mq;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * Kafka 日志消费者基类
 * <p>
 * 提供日志消费的基础设施，支持手动提交偏移量
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@RequiredArgsConstructor
public abstract class KafkaLogConsumer {

    protected final ObjectMapper objectMapper;

    /**
     * 解析日志消息
     */
    @SuppressWarnings("unchecked")
    protected Map<String, Object> parseLogEntry(String payload) {
        try {
            return objectMapper.readValue(payload, Map.class);
        } catch (Exception e) {
            log.error("解析日志消息失败: {}", e.getMessage());
            return null;
        }
    }

    /**
     * 提取字段值
     */
    protected <T> T extractField(Map<String, Object> logEntry, String field, Class<T> type) {
        Object value = logEntry.get(field);
        if (value == null) {
            return null;
        }
        return type.cast(value);
    }

    /**
     * 确认消息
     */
    protected void acknowledge(Acknowledgment acknowledgment) {
        if (acknowledgment != null) {
            acknowledgment.acknowledge();
        }
    }

    /**
     * 处理消费异常
     */
    protected void handleError(String topic, ConsumerRecord<String, String> record, Exception e) {
        log.error("Kafka 消息处理失败: topic={}, partition={}, offset={}, key={}",
                topic, record.partition(), record.offset(), record.key(), e);
    }
}
