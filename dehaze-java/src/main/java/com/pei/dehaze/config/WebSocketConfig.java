package com.pei.dehaze.config;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.SecurityConstants;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.messaging.Message;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.simp.config.ChannelRegistration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.messaging.simp.stomp.StompCommand;
import org.springframework.messaging.simp.stomp.StompHeaderAccessor;
import org.springframework.messaging.support.ChannelInterceptor;
import org.springframework.messaging.support.MessageHeaderAccessor;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;
import org.springframework.web.socket.server.HandshakeInterceptor;

import java.net.URI;
import java.util.Map;

/**
 * WebSocket 配置
 *
 * @author earthyzinc
 * @since 2.4.0
 */
@Configuration
@EnableWebSocketMessageBroker
@Slf4j
@RequiredArgsConstructor
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    private static final String ATTR_USER_ID = "userId";

    private final CorsConfig corsConfig;
    private final StringRedisTemplate stringRedisTemplate;

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        String[] allowedOrigins = corsConfig.getAllowedOrigins() != null
                ? corsConfig.getAllowedOrigins().toArray(new String[0])
                : new String[0];
        registry
                .addEndpoint("/ws")
                .setAllowedOriginPatterns(allowedOrigins)
                .addInterceptors(new SessionHandshakeInterceptor())
                .withSockJS();
        registry.addEndpoint("/ws-app")
                .setAllowedOriginPatterns(allowedOrigins)
                .addInterceptors(new SessionHandshakeInterceptor());
    }

    @Override
    public void configureMessageBroker(MessageBrokerRegistry registry) {
        registry.setApplicationDestinationPrefixes("/app");
        registry.enableSimpleBroker("/topic", "/queue");
        registry.setUserDestinationPrefix("/user");
    }

    @Override
    public void configureClientInboundChannel(ChannelRegistration registration) {
        registration.interceptors(new ChannelInterceptor() {
            @Override
            public Message<?> preSend(Message<?> message, MessageChannel channel) {
                StompHeaderAccessor accessor = MessageHeaderAccessor.getAccessor(message, StompHeaderAccessor.class);
                if (accessor == null || !StompCommand.CONNECT.equals(accessor.getCommand())) {
                    return ChannelInterceptor.super.preSend(message, channel);
                }

                Map<String, Object> sessionAttributes = accessor.getSessionAttributes();
                if (sessionAttributes == null) {
                    log.warn("WebSocket 连接被拒绝：会话属性缺失");
                    return null;
                }
                Object userIdObj = sessionAttributes.get(ATTR_USER_ID);
                if (userIdObj == null) {
                    log.warn("WebSocket 连接被拒绝：会话未认证");
                    return null;
                }
                String userId = String.valueOf(userIdObj);
                accessor.setUser(() -> userId);
                return message;
            }
        });
    }

    /**
     * 握手拦截器：从 URL query 或 Cookie 中提取 sessionId，查询 Redis 验证用户身份。
     * 对齐 Python/Go 的 Session ID 鉴权方式。
     */
    private class SessionHandshakeInterceptor implements HandshakeInterceptor {

        @Override
        public boolean beforeHandshake(ServerHttpRequest request, ServerHttpResponse response,
                                       WebSocketHandler wsHandler, Map<String, Object> attributes) {
            String sessionId = extractSessionId(request);
            if (CharSequenceUtil.isBlank(sessionId)) {
                log.warn("WebSocket 握手被拒绝：缺少 sessionId");
                return false;
            }
            String sessionJson = stringRedisTemplate.opsForValue()
                    .get(SecurityConstants.SESSION_PREFIX + sessionId);
            if (CharSequenceUtil.isBlank(sessionJson)) {
                log.warn("WebSocket 握手被拒绝：session 不存在或已过期");
                return false;
            }
            try {
                JSONObject session = JSONUtil.parseObj(sessionJson);
                Long userId = session.getLong("userId");
                if (userId == null) {
                    log.warn("WebSocket 握手被拒绝：session 缺少 userId");
                    return false;
                }
                attributes.put(ATTR_USER_ID, userId);
                return true;
            } catch (Exception e) {
                log.warn("WebSocket 握手被拒绝：session 解析失败", e);
                return false;
            }
        }

        @Override
        public void afterHandshake(ServerHttpRequest request, ServerHttpResponse response,
                                   WebSocketHandler wsHandler, Exception exception) {
        }

        private String extractSessionId(ServerHttpRequest request) {
            URI uri = request.getURI();
            String query = uri.getQuery();
            if (query != null) {
                for (String param : query.split("&")) {
                    String[] kv = param.split("=", 2);
                    if (kv.length == 2 && "sessionId".equals(kv[0])) {
                        return kv[1];
                    }
                }
            }
            for (String cookieHeader : request.getHeaders().getOrDefault("Cookie", java.util.Collections.emptyList())) {
                for (String cookie : cookieHeader.split(";")) {
                    String trimmed = cookie.trim();
                    int idx = trimmed.indexOf('=');
                    if (idx > 0 && SecurityConstants.SESSION_COOKIE_NAME.equals(trimmed.substring(0, idx))) {
                        return trimmed.substring(idx + 1);
                    }
                }
            }
            return null;
        }
    }
}
