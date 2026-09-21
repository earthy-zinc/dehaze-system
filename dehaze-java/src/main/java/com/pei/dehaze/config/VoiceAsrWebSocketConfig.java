package com.pei.dehaze.config;

import com.pei.dehaze.service.client.VoiceAsrProxyHandler;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.ServletWebSocketHandlerRegistry;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

/**
 * 语音流式 ASR 的 WebSocket 转发注册（{@code /ws/asr} → dehaze-python）。
 * <p>
 * 与 {@link WebSocketConfig}（STOMP 消息通道，{@code /ws}）是两个互不相干的映射，
 * 共享同一套容器 WebSocket 支持。
 *
 * @author earthyzinc
 * @since 2026-09-18
 */
@Configuration
@EnableWebSocket
@RequiredArgsConstructor
public class VoiceAsrWebSocketConfig implements WebSocketConfigurer {

    private final VoiceAsrProxyHandler voiceAsrProxyHandler;

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        // order 必须严格小于 STOMP 端点：@EnableWebSocketMessageBroker 为 /ws 注册了 SockJS 通配
        // /ws/**（order=1）。两者同序时由 bean 定义顺序决定谁命中，SockJS 抢到 /ws/asr 会按
        // 非法 transport 报错（SockJsUrlInfo 只认 xhr/websocket 等取值）
        ((ServletWebSocketHandlerRegistry) registry).setOrder(0);
        // 来源不设白名单：与 Go（CheckOrigin 恒 true）、python（不校验 Origin）一致，CORS 由网关统一处理
        registry.addHandler(voiceAsrProxyHandler, "/ws/asr");
    }
}
