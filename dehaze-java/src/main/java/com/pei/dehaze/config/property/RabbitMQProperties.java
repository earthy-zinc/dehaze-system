package com.pei.dehaze.config.property;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;

/**
 * RabbitMQ 配置属性
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Data
@ConfigurationProperties(prefix = "rabbitmq")
public class RabbitMQProperties {

    /**
     * 是否启用 RabbitMQ
     */
    private boolean enabled = false;

    /**
     * RabbitMQ 连接 URL (amqp://user:password@host:port/vhost)
     */
    private String url;

    /**
     * 交换机配置
     */
    private ExchangeProperty exchange = new ExchangeProperty();

    /**
     * 重连配置
     */
    private ReconnectProperty reconnect = new ReconnectProperty();

    /**
     * 交换机配置
     */
    @Data
    public static class ExchangeProperty {

        /**
         * 交换机名称
         */
        private String name = "dehaze.tasks";

        /**
         * 交换机类型 (direct/topic/fanout/headers)
         */
        private String type = "direct";

        /**
         * 路由键前缀
         */
        private String routingKeyPrefix = "task.";

    }

    /**
     * 重连配置
     */
    @Data
    public static class ReconnectProperty {

        /**
         * 最大重试次数 (0 表示无限重试)
         */
        private int maxRetries = 10;

        /**
         * 初始重连间隔
         */
        private Duration initialInterval = Duration.ofSeconds(1);

        /**
         * 最大重连间隔
         */
        private Duration maxInterval = Duration.ofSeconds(30);

        /**
         * 重连间隔乘数 (指数退避)
         */
        private double multiplier = 2.0;

    }
}
