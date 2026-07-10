package com.pei.dehaze.config.property;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.HashMap;
import java.util.Map;

/**
 * Kafka 配置属性
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Data
@ConfigurationProperties(prefix = "kafka")
public class KafkaProperties {

    /**
     * 是否启用 Kafka
     */
    private boolean enabled = false;

    /**
     * Kafka Broker 地址 (多个用逗号分隔)
     */
    private String bootstrapServers;

    /**
     * 生产者配置
     */
    private ProducerProperty producer = new ProducerProperty();

    /**
     * 消费者配置
     */
    private ConsumerProperty consumer = new ConsumerProperty();

    /**
     * 日志管道配置
     */
    private LogPipelineProperty logPipeline = new LogPipelineProperty();

    /**
     * 审计日志管道配置
     */
    private AuditPipelineProperty auditPipeline = new AuditPipelineProperty();

    /**
     * 生产者配置
     */
    @Data
    public static class ProducerProperty {

        /**
         * 消息确认模式 (all, 1, 0)
         */
        private String acks = "all";

        /**
         * 重试次数
         */
        private int retries = 3;

        /**
         * 批量发送大小 (字节)
         */
        private int batchSize = 16384;

        /**
         * 批量发送延迟 (毫秒)
         */
        private int lingerMs = 5;

        /**
         * 缓冲区大小 (字节)
         */
        private long bufferMemory = 33554432;

        /**
         * Key 序列化器
         */
        private String keySerializer = "org.apache.kafka.common.serialization.StringSerializer";

        /**
         * Value 序列化器
         */
        private String valueSerializer = "org.apache.kafka.common.serialization.StringSerializer";

    }

    /**
     * 消费者配置
     */
    @Data
    public static class ConsumerProperty {

        /**
         * 消费者组 ID
         */
        private String groupId = "dehaze-java";

        /**
         * 自动提交偏移量
         */
        private boolean enableAutoCommit = false;

        /**
         * 自动提交间隔 (毫秒)
         */
        private int autoCommitIntervalMs = 1000;

        /**
         * 消费起始位置 (earliest, latest)
         */
        private String autoOffsetReset = "earliest";

        /**
         * Key 反序列化器
         */
        private String keyDeserializer = "org.apache.kafka.common.serialization.StringDeserializer";

        /**
         * Value 反序列化器
         */
        private String valueDeserializer = "org.apache.kafka.common.serialization.StringDeserializer";

        /**
         * 单次拉取最大记录数
         */
        private int maxPollRecords = 100;

        /**
         * 拉取超时时间 (毫秒)
         */
        private int pollTimeoutMs = 1000;

    }

    /**
     * 日志管道配置
     */
    @Data
    public static class LogPipelineProperty {

        /**
         * 日志 Topic 名称
         */
        private String topic = "dehaze.logs";

        /**
         * 是否启用日志管道
         */
        private boolean enabled = false;

    }

    /**
     * 审计日志管道配置
     */
    @Data
    public static class AuditPipelineProperty {

        /**
         * 审计日志 Topic 名称
         */
        private String topic = "dehaze.audit";

        /**
         * 是否启用审计日志管道
         */
        private boolean enabled = false;

    }

    /**
     * 获取生产者配置 Map
     */
    public Map<String, Object> buildProducerProps() {
        Map<String, Object> props = new HashMap<>();
        if (bootstrapServers == null || bootstrapServers.isBlank()) {
            throw new IllegalStateException("Kafka bootstrapServers is not configured");
        }
        props.put("bootstrap.servers", bootstrapServers);
        props.put("acks", producer.getAcks());
        props.put("retries", producer.getRetries());
        props.put("batch.size", producer.getBatchSize());
        props.put("linger.ms", producer.getLingerMs());
        props.put("buffer.memory", producer.getBufferMemory());
        props.put("key.serializer", producer.getKeySerializer());
        props.put("value.serializer", producer.getValueSerializer());
        return props;
    }

    /**
     * 获取消费者配置 Map
     */
    public Map<String, Object> buildConsumerProps() {
        Map<String, Object> props = new HashMap<>();
        if (bootstrapServers == null || bootstrapServers.isBlank()) {
            throw new IllegalStateException("Kafka bootstrapServers is not configured");
        }
        props.put("bootstrap.servers", bootstrapServers);
        props.put("group.id", consumer.getGroupId());
        props.put("enable.auto.commit", consumer.isEnableAutoCommit());
        props.put("auto.commit.interval.ms", consumer.getAutoCommitIntervalMs());
        props.put("auto.offset.reset", consumer.getAutoOffsetReset());
        props.put("key.deserializer", consumer.getKeyDeserializer());
        props.put("value.deserializer", consumer.getValueDeserializer());
        props.put("max.poll.records", consumer.getMaxPollRecords());
        return props;
    }
}
