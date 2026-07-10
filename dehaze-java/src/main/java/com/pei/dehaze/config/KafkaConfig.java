package com.pei.dehaze.config;

import com.pei.dehaze.config.property.KafkaProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.annotation.EnableKafka;
import org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory;
import org.springframework.kafka.core.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Kafka 配置
 * <p>
 * 用于日志收集与流处理
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Configuration
@RequiredArgsConstructor
@EnableKafka
@EnableConfigurationProperties(KafkaProperties.class)
@ConditionalOnProperty(prefix = "kafka", name = "enabled", havingValue = "true")
public class KafkaConfig {

    private final KafkaProperties properties;

    /**
     * 生产者配置
     */
    @Bean
    public ProducerFactory<String, String> producerFactory() {
        Map<String, Object> props = properties.buildProducerProps();
        log.info("Kafka Producer 工厂已初始化: {}", properties.getBootstrapServers());
        return new DefaultKafkaProducerFactory<>(props);
    }

    /**
     * KafkaTemplate
     */
    @Bean
    public KafkaTemplate<String, String> kafkaTemplate(ProducerFactory<String, String> producerFactory) {
        return new KafkaTemplate<>(producerFactory);
    }

    /**
     * 消费者配置
     */
    @Bean
    public ConsumerFactory<String, String> consumerFactory() {
        Map<String, Object> props = properties.buildConsumerProps();
        return new DefaultKafkaConsumerFactory<>(props);
    }

    /**
     * 监听器容器工厂
     */
    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, String> kafkaListenerContainerFactory(
            ConsumerFactory<String, String> consumerFactory) {
        ConcurrentKafkaListenerContainerFactory<String, String> factory = new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(consumerFactory);
        factory.setConcurrency(3);
        factory.getContainerProperties().setPollTimeout(properties.getConsumer().getPollTimeoutMs());
        factory.getContainerProperties().setAckMode(
                org.springframework.kafka.listener.ContainerProperties.AckMode.MANUAL_IMMEDIATE);
        return factory;
    }

    /**
     * 日志 Topic (日志收集管道)
     */
    @Bean
    public NewTopic logTopic() {
        return new NewTopic(properties.getLogPipeline().getTopic(), 3, (short) 1);
    }

    /**
     * 审计日志 Topic
     */
    @Bean
    public NewTopic auditLogTopic() {
        return new NewTopic(properties.getAuditPipeline().getTopic(), 3, (short) 1);
    }
}
