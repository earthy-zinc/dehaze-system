package com.pei.dehaze.config;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.data.redis.connection.Message;
import org.springframework.data.redis.connection.MessageListener;
import org.springframework.data.redis.listener.ChannelTopic;
import org.springframework.data.redis.listener.RedisMessageListenerContainer;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;

/**
 * 缓存失效广播中继
 *
 * <p>订阅 Redis Pub/Sub 频道 cache:invalidation，当其他实例主动失效缓存（@CacheEvict）时，
 * 本实例收到消息后清除本地 L1 Caffeine 缓存，避免多实例间 L1 不一致导致脏读。
 *
 * <p>senderId 用于忽略自己发送的消息，与 MultiLevelCache 发布时使用的 senderId 一致。
 *
 * @author earthyzinc
 * @since 2026-07-28
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "spring.cache.enabled", havingValue = "true")
public class CacheInvalidationRelay implements MessageListener {

    private final RedisMessageListenerContainer redisMessageListenerContainer;
    private final MultiLevelCacheManager cacheManager;

    @PostConstruct
    public void init() {
        redisMessageListenerContainer.addMessageListener(this,
                new ChannelTopic(MultiLevelCache.CACHE_INVALIDATION_CHANNEL));
        log.info("缓存失效广播已订阅频道: {}", MultiLevelCache.CACHE_INVALIDATION_CHANNEL);
    }

    @Override
    public void onMessage(Message message, byte[] pattern) {
        try {
            String body = new String(message.getBody(), StandardCharsets.UTF_8);
            JSONObject msg = JSONUtil.parseObj(body);
            String cacheName = msg.getStr("type");
            String msgSenderId = msg.getStr("senderId");
            if (cacheName == null || msgSenderId == null) {
                return;
            }
            if (msgSenderId.equals(cacheManager.getSenderId())) {
                return;
            }
            org.springframework.cache.Cache cache = cacheManager.getCacheIfPresent(cacheName);
            if (cache instanceof MultiLevelCache mlc) {
                mlc.clearLocal();
            }
        } catch (Exception e) {
            log.warn("处理缓存失效消息失败", e);
        }
    }
}
