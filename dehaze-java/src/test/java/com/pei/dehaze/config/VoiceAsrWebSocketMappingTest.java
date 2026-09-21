package com.pei.dehaze.config;

import com.pei.dehaze.config.property.AlgorithmProperties;
import com.pei.dehaze.service.client.VoiceAsrProxyHandler;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.AnnotationAwareOrderComparator;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockServletContext;
import org.springframework.web.context.support.AnnotationConfigWebApplicationContext;
import org.springframework.web.servlet.HandlerMapping;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;
import org.springframework.web.socket.server.support.WebSocketHandlerMapping;
import org.springframework.web.servlet.handler.SimpleUrlHandlerMapping;

import java.net.http.HttpClient;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@code /ws/asr} 的映射归属与优先级守卫。
 * <p>
 * STOMP 端点为 {@code /ws} 注册的是 SockJS 通配 {@code /ws/**}，它同样匹配 {@code /ws/asr}；
 * 两个映射默认 order 相同时命中由 bean 定义顺序决定（不确定），SockJS 抢到后会按非法 transport
 * 报错。本测试锁定"裸 handler 映射 order 严格更小"，即真实 DispatcherServlet 解析顺序下
 * {@code /ws/asr} 必由语音转发 handler 承接。
 *
 * @author earthyzinc
 * @since 2026-09-18
 */
@DisplayName("/ws/asr 映射优先级测试")
class VoiceAsrWebSocketMappingTest {

    @Test
    @DisplayName("/ws/asr 归属裸 handler，且 order 严格小于 STOMP 的 SockJS 通配")
    void asrMapping_takesPrecedenceOverStompWildcard() throws Exception {
        try (AnnotationConfigWebApplicationContext context = new AnnotationConfigWebApplicationContext()) {
            context.setServletContext(new MockServletContext());
            // 同时注册两套 WebSocket 配置：refresh 成功本身即验证 @EnableWebSocket 与
            // @EnableWebSocketMessageBroker 共存无 bean 冲突（无同名 bean / 无覆盖）
            context.register(TestWebConfig.class, VoiceAsrWebSocketConfig.class);
            context.refresh();

            List<HandlerMapping> mappings = context.getBeansOfType(HandlerMapping.class).values().stream()
                    .sorted(AnnotationAwareOrderComparator.INSTANCE)
                    .toList();

            SimpleUrlHandlerMapping voiceMapping = mappingOwning(mappings, "/ws/asr");
            assertThat(voiceMapping)
                    .as("/ws/asr 必须注册在 WebSocketHandlerMapping 上（VoiceAsrWebSocketConfig）")
                    .isNotNull();
            SimpleUrlHandlerMapping stompMapping = mappingOwning(mappings, "/ws/**");
            assertThat(stompMapping)
                    .as("STOMP 的 SockJS 端点应注册 /ws/** 通配（本测试的前提）")
                    .isNotNull();
            assertThat(voiceMapping.getOrder())
                    .as("order 必须严格小于 SockJS 通配所在映射，否则命中由 bean 顺序决定")
                    .isLessThan(stompMapping.getOrder());

            // 按真实解析顺序（order 升序）取第一个能处理 /ws/asr 的映射：必须是语音转发 handler
            HandlerMapping firstMatch = null;
            for (HandlerMapping mapping : mappings) {
                if (mapping.getHandler(new MockHttpServletRequest("GET", "/ws/asr")) != null) {
                    firstMatch = mapping;
                    break;
                }
            }
            assertThat(firstMatch)
                    .as("DispatcherServlet 解析 /ws/asr 时命中的第一个映射")
                    .isSameAs(voiceMapping);
        }
    }

    /** 返回 urlMap 中登记了该模式的映射（仅 SimpleUrlHandlerMapping 才有 urlMap 与 order） */
    private static SimpleUrlHandlerMapping mappingOwning(List<HandlerMapping> mappings, String pattern) {
        for (HandlerMapping mapping : mappings) {
            if (mapping instanceof SimpleUrlHandlerMapping urlMapping
                    && urlMapping.getUrlMap() != null
                    && urlMapping.getUrlMap().containsKey(pattern)) {
                return urlMapping;
            }
        }
        return null;
    }

    @Configuration
    @EnableWebMvc
    @EnableWebSocketMessageBroker
    static class TestWebConfig implements WebSocketMessageBrokerConfigurer {

        @Bean
        VoiceAsrProxyHandler voiceAsrProxyHandler() {
            return new VoiceAsrProxyHandler(new AlgorithmProperties(), HttpClient.newHttpClient());
        }

        @Override
        public void registerStompEndpoints(StompEndpointRegistry registry) {
            // 与生产 WebSocketConfig 同形：/ws 带 SockJS（注册通配 /ws/**），/ws-app 不带
            registry.addEndpoint("/ws").withSockJS();
            registry.addEndpoint("/ws-app");
        }

        @Override
        public void configureMessageBroker(MessageBrokerRegistry registry) {
            registry.setApplicationDestinationPrefixes("/app");
            registry.enableSimpleBroker("/topic", "/queue");
            registry.setUserDestinationPrefix("/user");
        }
    }

    /** 断言用：确认 voice 映射确实是 WebSocketHandlerMapping 而非其它类型 */
    @Test
    @DisplayName("语音映射类型为 WebSocketHandlerMapping（裸 WebSocketHandler 路由）")
    void asrMapping_isWebSocketHandlerMapping() {
        try (AnnotationConfigWebApplicationContext context = new AnnotationConfigWebApplicationContext()) {
            context.setServletContext(new MockServletContext());
            context.register(TestWebConfig.class, VoiceAsrWebSocketConfig.class);
            context.refresh();
            assertThat(mappingOwning(context.getBeansOfType(HandlerMapping.class).values().stream().toList(), "/ws/asr"))
                    .isInstanceOf(WebSocketHandlerMapping.class);
        }
    }
}
