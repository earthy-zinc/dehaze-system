package com.pei.dehaze.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.boot.autoconfigure.cache.CacheProperties;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cache.CacheManager;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.data.redis.cache.RedisCacheConfiguration;
import org.springframework.data.redis.cache.RedisCacheManager;
import org.springframework.data.redis.cache.RedisCacheWriter;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.serializer.GenericJackson2JsonRedisSerializer;
import org.springframework.data.redis.serializer.RedisSerializationContext;
import org.springframework.data.redis.serializer.RedisSerializer;

import java.time.Duration;

/**
 * 缓存配置（L1 Caffeine + L2 Redis 多级缓存）
 *
 * <p>多级缓存架构：
 * <ul>
 *   <li>L1：Caffeine 本地缓存，TTL 5 分钟，防热 key 击穿 Redis</li>
 *   <li>L2：Redis 分布式缓存，TTL 1 小时</li>
 *   <li>SingleFlight：防缓存击穿（热点 key 失效瞬间合并并发回源）</li>
 *   <li>空值缓存：allowNullValues=true，防缓存穿透</li>
 *   <li>Prometheus 指标：dehaze_cache_hits_total / dehaze_cache_misses_total / dehaze_cache_loader_total</li>
 * </ul>
 *
 * @author earthyzinc
 * @since 2023/12/4
 */
@EnableCaching
@EnableConfigurationProperties(CacheProperties.class)
@Configuration
@ConditionalOnProperty(name = "spring.cache.enabled", havingValue = "true", matchIfMissing = false)
public class RedisCacheConfig {

    /**
     * L2 Redis CacheManager（不暴露为 @Primary，仅作为多级缓存的 L2 后端）
     */
    @Bean
    public RedisCacheManager redisCacheManager(RedisConnectionFactory redisConnectionFactory, CacheProperties cacheProperties) {
        // 允许缓存 null 值（防穿透），通过 cacheConfiguration 的 disableCachingNullValues() 控制
        return RedisCacheManager.builder(RedisCacheWriter.nonLockingRedisCacheWriter(redisConnectionFactory))
                .cacheDefaults(redisCacheConfiguration(cacheProperties))
                .build();
    }

    /**
     * 多级缓存管理器（@Primary，Spring Cache 注解使用此管理器）
     */
    @Bean
    @Primary
    public CacheManager cacheManager(RedisCacheManager redisCacheManager, MeterRegistry meterRegistry) {
        return new MultiLevelCacheManager(redisCacheManager, meterRegistry, Duration.ofMinutes(5));
    }

    /**
     * 自定义 RedisCacheConfiguration
     */
    @Bean
    RedisCacheConfiguration redisCacheConfiguration(CacheProperties cacheProperties) {
        RedisCacheConfiguration config = RedisCacheConfiguration.defaultCacheConfig();

        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
        objectMapper.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        objectMapper.activateDefaultTyping(
                objectMapper.getPolymorphicTypeValidator(),
                ObjectMapper.DefaultTyping.NON_FINAL
        );
        GenericJackson2JsonRedisSerializer jsonSerializer = new GenericJackson2JsonRedisSerializer(objectMapper);

        config = config.serializeKeysWith(RedisSerializationContext.SerializationPair.fromSerializer(RedisSerializer.string()));
        config = config.serializeValuesWith(RedisSerializationContext.SerializationPair.fromSerializer(jsonSerializer));

        CacheProperties.Redis redisProperties = cacheProperties.getRedis();

        if (redisProperties.getTimeToLive() != null) {
            config = config.entryTtl(redisProperties.getTimeToLive());
        }
        // 允许缓存 null 值（防穿透），不调用 disableCachingNullValues
        if (!redisProperties.isUseKeyPrefix()) {
            config = config.disableKeyPrefix();
        }
        config = config.computePrefixWith(name -> name + ":");
        return config;
    }
}
