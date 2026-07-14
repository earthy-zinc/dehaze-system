package com.pei.dehaze.config;

import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.cache.Cache;
import org.springframework.cache.CacheManager;

import java.time.Duration;
import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * 多级缓存管理器（L1 Caffeine + L2 Redis）
 *
 * <p>为每个 cacheName 创建一个 MultiLevelCache 实例，
 * L2 委托给已有的 RedisCacheManager。
 */
public class MultiLevelCacheManager implements CacheManager {

    private final CacheManager l2CacheManager;
    private final MeterRegistry meterRegistry;
    private final Duration l1Expire;
    private final BloomFilterService bloomFilterService;
    private final ConcurrentMap<String, Cache> cacheMap = new ConcurrentHashMap<>();

    public MultiLevelCacheManager(CacheManager l2CacheManager, MeterRegistry meterRegistry, Duration l1Expire) {
        this(l2CacheManager, meterRegistry, l1Expire, null);
    }

    public MultiLevelCacheManager(CacheManager l2CacheManager, MeterRegistry meterRegistry, Duration l1Expire,
                                  BloomFilterService bloomFilterService) {
        this.l2CacheManager = l2CacheManager;
        this.meterRegistry = meterRegistry;
        this.l1Expire = l1Expire;
        this.bloomFilterService = bloomFilterService;
    }

    @Override
    public Cache getCache(String name) {
        return cacheMap.computeIfAbsent(name, n -> {
            Cache l2Cache = l2CacheManager.getCache(n);
            if (l2Cache == null) {
                return null;
            }
            return new MultiLevelCache(n, l1Expire, l2Cache, meterRegistry, bloomFilterService);
        });
    }

    @Override
    public Collection<String> getCacheNames() {
        return l2CacheManager.getCacheNames();
    }
}
