package com.pei.dehaze.module.iot.plugin.emqx.config;

import cn.hutool.core.util.IdUtil;
import com.pei.dehaze.module.iot.api.device.IotDeviceUpstreamApi;
import com.pei.dehaze.module.iot.plugin.common.downstream.IotDeviceDownstreamHandler;
import com.pei.dehaze.module.iot.plugin.emqx.downstream.IotDeviceDownstreamHandlerImpl;
import com.pei.dehaze.module.iot.plugin.emqx.upstream.IotDeviceUpstreamServer;
import io.vertx.core.Vertx;
import io.vertx.mqtt.MqttClient;
import io.vertx.mqtt.MqttClientOptions;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * IoT 插件 EMQX 的专用自动配置类
 *
 * @author haohao
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(IotPluginEmqxProperties.class)
public class IotPluginEmqxAutoConfiguration {

    @Bean
    public Vertx vertx() {
        return Vertx.vertx();
    }

    @Bean
    public MqttClient mqttClient(Vertx vertx, IotPluginEmqxProperties emqxProperties) {
        MqttClientOptions options = new MqttClientOptions()
                .setClientId("pei-iot-downstream-" + IdUtil.fastSimpleUUID())
                .setUsername(emqxProperties.getMqttUsername())
                .setPassword(emqxProperties.getMqttPassword())
                .setSsl(emqxProperties.getMqttSsl());
        return MqttClient.create(vertx, options);
    }

    @Bean(initMethod = "start", destroyMethod = "stop")
    public IotDeviceUpstreamServer deviceUpstreamServer(IotDeviceUpstreamApi deviceUpstreamApi,
                                                        IotPluginEmqxProperties emqxProperties,
                                                        Vertx vertx,
                                                        MqttClient mqttClient) {
        return new IotDeviceUpstreamServer(emqxProperties, deviceUpstreamApi, vertx, mqttClient);
    }

    @Bean
    public IotDeviceDownstreamHandler deviceDownstreamHandler(MqttClient mqttClient) {
        return new IotDeviceDownstreamHandlerImpl(mqttClient);
    }

}
