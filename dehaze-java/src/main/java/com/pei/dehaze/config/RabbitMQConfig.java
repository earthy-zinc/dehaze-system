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
                .withArgument("x-message-ttl", 86400000) // 24小时 TTL
                .withArgument("x-dead-letter-exchange", properties.getExchange().getName() + ".dlx")
                .withArgument("x-dead-letter-routing-key", resolveRoutingKey("export.dlx"))
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
        return BindingBuilder.bind(exportQueue).to(taskExchange).with(resolveRoutingKey("export"));
    }

    /**
     * 绑定导出死信队列到死信交换机
     */
    @Bean
    public Binding exportDlxBinding(Queue exportDlxQueue, DirectExchange taskDlxExchange) {
        return BindingBuilder.bind(exportDlxQueue).to(taskDlxExchange).with(resolveRoutingKey("export.dlx"));
    }

    /**
     * 解析路由键
     */
    private String resolveRoutingKey(String queueName) {
        String prefix = properties.getExchange().getRoutingKeyPrefix();
        return prefix + queueName;
    }
}
