package com.pei.dehaze.config;

import cn.hutool.json.JSONObject;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.Cache;
import org.springframework.cache.support.AbstractValueAdaptingCache;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.time.Duration;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

import com.github.benmanes.caffeine.cache.Caffeine;

/**
 * 多级缓存（L1 Caffeine + L2 Redis）
 *
 * <p>读流程：L1 -> L2 -> 回源（由 Spring Cache AOP 触发）
 * <p>写流程：同时写入 L1 和 L2
 * <p>SingleFlight：相同 key 的并发回源请求合并，防缓存击穿
 * <p>空值缓存：通过 AbstractValueAdaptingCache 的 allowNullValues=true 机制缓存 null
 * <p>多实例失效广播：evict/clear 时通过 Redis Pub/Sub（cache:invalidation 频道）通知其他实例清 L1
 */
@Slf4j
public class MultiLevelCache extends AbstractValueAdaptingCache {

    static final String CACHE_INVALIDATION_CHANNEL = "cache:invalidation";

    private final String name;
    private final com.github.benmanes.caffeine.cache.Cache<Object, Object> l1Cache;
    private final Cache l2Cache;
    private final StringRedisTemplate publisher;
    private final String senderId;

    // SingleFlight：相同 key 的并发回源合并
    private final ConcurrentHashMap<Object, CompletableFuture<Object>> inFlight = new ConcurrentHashMap<>();

    // Prometheus 指标
    private final Counter l1HitCounter;
    private final Counter l1MissCounter;
    private final Counter l2HitCounter;
    private final Counter l2MissCounter;
    private final Counter loaderCounter;

    public MultiLevelCache(
            String name,
            Duration l1Expire,
            Cache l2Cache,
            MeterRegistry meterRegistry,
            StringRedisTemplate publisher,
            String senderId) {
        super(true); // allowNullValues=true，启用空值缓存防穿透
        this.name = name;
        this.l2Cache = l2Cache;
        this.publisher = publisher;
        this.senderId = senderId;
        this.l1Cache = Caffeine.newBuilder()
                .expireAfterWrite(l1Expire)
                .maximumSize(1000)
                .build();

        Tags nameTag = Tags.of("cache", name);
        this.l1HitCounter = meterRegistry.counter("dehaze_cache_hits_total", nameTag.and("layer", "L1"));
        this.l1MissCounter = meterRegistry.counter("dehaze_cache_misses_total", nameTag.and("layer", "L1"));
        this.l2HitCounter = meterRegistry.counter("dehaze_cache_hits_total", nameTag.and("layer", "L2"));
        this.l2MissCounter = meterRegistry.counter("dehaze_cache_misses_total", nameTag.and("layer", "L2"));
        this.loaderCounter = meterRegistry.counter("dehaze_cache_loader_total", nameTag.and("result", "hit"));
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public Object getNativeCache() {
        return this;
    }

    @Override
    protected Object lookup(Object key) {
        // 1. L1 Caffeine
        Object l1Value = l1Cache.getIfPresent(key);
        if (l1Value != null) {
            l1HitCounter.increment();
            return l1Value;
        }
        l1MissCounter.increment();

        // 2. L2 Redis（使用 get(key) 返回 ValueWrapper，不触发 loader）
        // 包裹 try/catch：L2 中可能存在因 schema 变化或类型信息不匹配而无法反序列化的脏数据，
        // 此时 evict 该 key（清 L1+L2）并返回 null，让上层 CacheInterceptor 视为未命中回源查库，
        // 而不是把反序列化异常抛上去被误判为业务错误。
        org.springframework.cache.Cache.ValueWrapper l2Wrapper;
        try {
            l2Wrapper = l2Cache.get(key);
        } catch (RuntimeException e) {
            log.warn("L2 Redis 反序列化失败，清除脏缓存并回源: cache={}, key={}, error={}", name, key, e.getMessage());
            evict(key);
            return null;
        }
        if (l2Wrapper != null) {
            Object l2Value = l2Wrapper.get();
            // L2 缓存了 null 值（防穿透场景）：ValueWrapper 非 null 但 get() 返回 null。
            // 需转换为 store value 哨兵，否则上层 get() 无法区分"缓存命中 null"与"未命中"。
            if (l2Value == null) {
                l2Value = toStoreValue(null);
            }
            l2HitCounter.increment();
            // 回填 L1（包括 null 值哨兵，防穿透）
            l1Cache.put(key, l2Value);
            return l2Value;
        }
        l2MissCounter.increment();

        return null;
    }

    @Override
    public <T> T get(Object key, Callable<T> valueLoader) {
        // 先查缓存（lookup 方法）
        Object cached = lookup(key);
        if (cached != null) {
            @SuppressWarnings("unchecked")
            T value = (T) fromStoreValue(cached);
            return value;
        }

        // SingleFlight：合并并发回源
        CompletableFuture<Object> future = inFlight.computeIfAbsent(key, k ->
                CompletableFuture.supplyAsync(() -> {
                    try {
                        Object result = valueLoader.call();
                        return toStoreValue(result);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }));

        try {
            Object storeValue = future.get();
            // 写入 L1 + L2
            if (storeValue != null) {
                l1Cache.put(key, storeValue);
                l2Cache.put(key, storeValue);
            }
            loaderCounter.increment();
            @SuppressWarnings("unchecked")
            T result = (T) fromStoreValue(storeValue);
            return result;
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            inFlight.remove(key);
        }
    }

    @Override
    public void put(Object key, Object value) {
        Object storeValue = toStoreValue(value);
        // 同时写入 L1 和 L2
        l1Cache.put(key, storeValue);
        l2Cache.put(key, storeValue);
    }

    @Override
    public void evict(Object key) {
        // Cache-Aside：先删 L2 再删 L1
        l2Cache.evict(key);
        l1Cache.invalidate(key);
        // 广播失效消息，通知其他实例清 L1
        publishInvalidation(key);
    }

    @Override
    public void clear() {
        l2Cache.clear();
        l1Cache.invalidateAll();
        publishInvalidation(null);
    }

    void clearLocal() {
        l1Cache.invalidateAll();
    }

    private void publishInvalidation(Object key) {
        if (publisher == null) {
            return;
        }
        try {
            JSONObject msg = new JSONObject();
            msg.set("type", name);
            msg.set("key", key != null ? key.toString() : null);
            msg.set("senderId", senderId);
            publisher.convertAndSend(CACHE_INVALIDATION_CHANNEL, msg.toString());
        } catch (Exception e) {
            log.warn("发布缓存失效消息失败（不影响本地失效）: cache={}, key={}", name, key, e);
        }
    }
}
