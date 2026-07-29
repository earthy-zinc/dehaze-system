package com.pei.dehaze.config;

import com.pei.dehaze.config.property.RabbitMQProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.*;
import org.springframework.amqp.rabbit.config.SimpleRabbitListenerContainerFactory;
import org.springframework.amqp.rabbit.connection.CachingConnectionFactory;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.amqp.support.converter.MessageConverter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.amqp.core.AcknowledgeMode;

import java.net.URI;
import java.net.URISyntaxException;

/**
 * RabbitMQ 配置
 * <p>
 * 用于异步任务分发（导出、批量操作等）
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Configuration
@RequiredArgsConstructor
@EnableConfigurationProperties(RabbitMQProperties.class)
@ConditionalOnProperty(prefix = "rabbitmq", name = "enabled", havingValue = "true")
public class RabbitMQConfig {

    private final RabbitMQProperties properties;

    /**
     * RabbitMQ 连接工厂
     */
    @Bean
    @Primary
    public ConnectionFactory rabbitConnectionFactory() {
        CachingConnectionFactory factory = new CachingConnectionFactory();
        try {
            factory.setUri(new URI(properties.getUrl()));
        } catch (URISyntaxException e) {
            throw new IllegalArgumentException("Invalid RabbitMQ URI: " + properties.getUrl(), e);
        }
        factory.setPublisherConfirmType(CachingConnectionFactory.ConfirmType.CORRELATED);
        factory.setPublisherReturns(true);
        log.info("RabbitMQ 连接工厂已初始化: {}", properties.getUrl());
        return factory;
    }

    /**
     * JSON 消息转换器
     */
    @Bean
    public MessageConverter jsonMessageConverter() {
        return new Jackson2JsonMessageConverter();
    }

    /**
     * RabbitTemplate 配置
     */
    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory,
            MessageConverter jsonMessageConverter) {
        RabbitTemplate template = new RabbitTemplate(connectionFactory);
        template.setMessageConverter(jsonMessageConverter);
        template.setMandatory(true);

        // 消息发送确认回调
        template.setConfirmCallback((correlationData, ack, cause) -> {
            if (!ack) {
                log.error("RabbitMQ 消息发送失败: {}", cause);
            }
        });

        // 消息退回回调
        template.setReturnsCallback(returned -> {
            log.error("RabbitMQ 消息被退回: {}, replyCode: {}, replyText: {}",
                    returned.getMessage(), returned.getReplyCode(), returned.getReplyText());
        });

        return template;
    }

    /**
     * 监听器容器工厂
     */
    @Bean
    public SimpleRabbitListenerContainerFactory rabbitListenerContainerFactory(
            ConnectionFactory connectionFactory, MessageConverter jsonMessageConverter) {
        SimpleRabbitListenerContainerFactory factory = new SimpleRabbitListenerContainerFactory();
        factory.setConnectionFactory(connectionFactory);
        factory.setMessageConverter(jsonMessageConverter);
        factory.setConcurrentConsumers(3);
        factory.setMaxConcurrentConsumers(10);
        factory.setPrefetchCount(10);
        factory.setAcknowledgeMode(AcknowledgeMode.MANUAL);
        factory.setDefaultRequeueRejected(false);
        return factory;
    }

    /**
     * 任务交换机 (Direct 类型)
     */
    @Bean
    public DirectExchange taskExchange() {
        return ExchangeBuilder
                .directExchange(properties.getExchange().getName())
                .durable(true)
                .build();
    }

    /**
     * 死信交换机
     */
    @Bean
    public DirectExchange taskDlxExchange() {
        return ExchangeBuilder
                .directExchange(properties.getExchange().getName() + ".dlx")
                .durable(true)
                .build();
    }

    /**
     * 导出任务队列（配置 DLX：过期或 reject 的消息进入死信队列）
     */
    @Bean
    public Queue exportQueue() {
        return QueueBuilder
                .durable("task.export")
                .withArgument("x-message-ttl", 86400000)
                .withArgument("x-dead-letter-exchange", properties.getExchange().getName() + ".dlx")
                .withArgument("x-dead-letter-routing-key", "task.export.dlx")
                .build();
    }

    @Bean
    public Queue exportRetryQueue0() {
        return QueueBuilder
                .durable("task.export.retry.0")
                .withArgument("x-message-ttl", 5000)
                .withArgument("x-dead-letter-exchange", properties.getExchange().getName())
                .withArgument("x-dead-letter-routing-key", "task.export")
                .build();
    }

    @Bean
    public Queue exportRetryQueue1() {
        return QueueBuilder
                .durable("task.export.retry.1")
                .withArgument("x-message-ttl", 30000)
                .withArgument("x-dead-letter-exchange", properties.getExchange().getName())
                .withArgument("x-dead-letter-routing-key", "task.export")
                .build();
    }

    @Bean
    public Queue exportRetryQueue2() {
        return QueueBuilder
                .durable("task.export.retry.2")
                .withArgument("x-message-ttl", 300000)
                .withArgument("x-dead-letter-exchange", properties.getExchange().getName())
                .withArgument("x-dead-letter-routing-key", "task.export")
                .build();
    }

    /**
     * 导出任务死信队列
     */
    @Bean
    public Queue exportDlxQueue() {
        return QueueBuilder.durable("task.export.dlx").build();
    }

    /**
     * 绑定导出队列到交换机
     */
    @Bean
    public Binding exportBinding(Queue exportQueue, DirectExchange taskExchange) {
        return BindingBuilder.bind(exportQueue).to(taskExchange).with("task.export");
    }

    @Bean
    public Binding exportRetryBinding0(Queue exportRetryQueue0, DirectExchange taskExchange) {
        return BindingBuilder.bind(exportRetryQueue0).to(taskExchange).with("task.export.retry.0");
    }

    @Bean
    public Binding exportRetryBinding1(Queue exportRetryQueue1, DirectExchange taskExchange) {
        return BindingBuilder.bind(exportRetryQueue1).to(taskExchange).with("task.export.retry.1");
    }

    @Bean
    public Binding exportRetryBinding2(Queue exportRetryQueue2, DirectExchange taskExchange) {
        return BindingBuilder.bind(exportRetryQueue2).to(taskExchange).with("task.export.retry.2");
    }

    @Bean
    public Binding exportDlxBinding(Queue exportDlxQueue, DirectExchange taskDlxExchange) {
        return BindingBuilder.bind(exportDlxQueue).to(taskDlxExchange).with("task.export.dlx");
    }

    /**
     * 低分告警队列（配置 DLX：过期或 reject 的消息进入死信队列）
     */
    @Bean
    public Queue lowRatingAlertQueue() {
        return QueueBuilder
                .durable("feedback.low_rating")
                .withArgument("x-message-ttl", 86400000)
                .withArgument("x-dead-letter-exchange", properties.getExchange().getName() + ".dlx")
                .withArgument("x-dead-letter-routing-key", "feedback.low_rating.dlx")
                .build();
    }

    @Bean
    public Queue lowRatingAlertRetryQueue0() {
        return QueueBuilder
                .durable("feedback.low_rating.retry.0")
                .withArgument("x-message-ttl", 5000)
                .withArgument("x-dead-letter-exchange", properties.getExchange().getName())
                .withArgument("x-dead-letter-routing-key", "feedback.low_rating")
                .build();
    }

    @Bean
    public Queue lowRatingAlertRetryQueue1() {
        return QueueBuilder
                .durable("feedback.low_rating.retry.1")
                .withArgument("x-message-ttl", 30000)
                .withArgument("x-dead-letter-exchange", properties.getExchange().getName())
                .withArgument("x-dead-letter-routing-key", "feedback.low_rating")
                .build();
    }

    @Bean
    public Queue lowRatingAlertRetryQueue2() {
        return QueueBuilder
                .durable("feedback.low_rating.retry.2")
                .withArgument("x-message-ttl", 300000)
                .withArgument("x-dead-letter-exchange", properties.getExchange().getName())
                .withArgument("x-dead-letter-routing-key", "feedback.low_rating")
                .build();
    }

    /**
     * 低分告警死信队列
     */
    @Bean
    public Queue lowRatingAlertDlxQueue() {
        return QueueBuilder.durable("feedback.low_rating.dlx").build();
    }

    /**
     * 绑定低分告警队列到交换机
     */
    @Bean
    public Binding lowRatingAlertBinding(Queue lowRatingAlertQueue, DirectExchange taskExchange) {
        return BindingBuilder.bind(lowRatingAlertQueue).to(taskExchange).with("feedback.low_rating");
    }

    @Bean
    public Binding lowRatingAlertRetryBinding0(Queue lowRatingAlertRetryQueue0, DirectExchange taskExchange) {
        return BindingBuilder.bind(lowRatingAlertRetryQueue0).to(taskExchange).with("feedback.low_rating.retry.0");
    }

    @Bean
    public Binding lowRatingAlertRetryBinding1(Queue lowRatingAlertRetryQueue1, DirectExchange taskExchange) {
        return BindingBuilder.bind(lowRatingAlertRetryQueue1).to(taskExchange).with("feedback.low_rating.retry.1");
    }

    @Bean
    public Binding lowRatingAlertRetryBinding2(Queue lowRatingAlertRetryQueue2, DirectExchange taskExchange) {
        return BindingBuilder.bind(lowRatingAlertRetryQueue2).to(taskExchange).with("feedback.low_rating.retry.2");
    }

    @Bean
    public Binding lowRatingAlertDlxBinding(Queue lowRatingAlertDlxQueue, DirectExchange taskDlxExchange) {
        return BindingBuilder.bind(lowRatingAlertDlxQueue).to(taskDlxExchange).with("feedback.low_rating.dlx");
    }

    private String resolveRoutingKey(String queueName) {
        return queueName;
    }
}
