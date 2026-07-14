package com.pei.dehaze.config;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBloomFilter;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Component;

import java.util.Collection;

/**
 * 布隆过滤器服务（基于 Redisson RBloomFilter）
 *
 * <p>用于缓存穿透防护：在查询缓存前预判 key 是否可能存在，
 * 若布隆过滤器判定"不存在"则直接返回 null，避免回源 DB。
 *
 * <p>使用前必须调用 {@link #init} 初始化（导入全量已有 key），
 * 否则 {@link #mightContain} 始终返回 true（安全降级，放行所有请求）。
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class BloomFilterService {

    private static final long DEFAULT_EXPECTED_INSERTIONS = 10_000L;
    private static final double DEFAULT_FALSE_PROBABILITY = 0.01;

    private final RedissonClient redissonClient;

    /**
     * 初始化布隆过滤器（重建：先删后建，再灌入全量 key）
     *
     * <p>典型调用时机：应用启动时由业务 Service 调用，导入该缓存空间的全量已有 key。
     *
     * @param name              过滤器名称（通常与 cacheName 对齐）
     * @param keys              全量已有 key
     * @param expectedInsertions 预期元素数量
     * @param falseProbability  误判率
     */
    public void init(String name, Collection<String> keys, long expectedInsertions, double falseProbability) {
        RBloomFilter<String> filter = getFilter(name);
        filter.delete();
        filter.tryInit(expectedInsertions, falseProbability);
        if (keys != null) {
            keys.forEach(filter::add);
        }
        log.info("布隆过滤器 [{}] 初始化完成: 元素数={}, 预期容量={}, 误判率={}",
                name, keys != null ? keys.size() : 0, expectedInsertions, falseProbability);
    }

    /**
     * 使用默认容量和误判率初始化布隆过滤器
     */
    public void init(String name, Collection<String> keys) {
        init(name, keys, DEFAULT_EXPECTED_INSERTIONS, DEFAULT_FALSE_PROBABILITY);
    }

    /**
     * 向布隆过滤器添加 key（新增数据时调用）
     */
    public void add(String name, String key) {
        if (!isInitialized(name)) {
            return;
        }
        getFilter(name).add(key);
    }

    /**
     * 判断 key 是否可能存在
     *
     * <p>安全降级：若过滤器未初始化，返回 true（放行，不阻断查询）。
     * 仅当过滤器已初始化且判定"不存在"时返回 false（阻断查询，防穿透）。
     */
    public boolean mightContain(String name, String key) {
        if (!isInitialized(name)) {
            return true;
        }
        return getFilter(name).contains(key);
    }

    /**
     * 判断布隆过滤器是否已初始化（Redis 中存在对应的 key）
     */
    public boolean isInitialized(String name) {
        return getFilter(name).isExists();
    }

    private RBloomFilter<String> getFilter(String name) {
        return redissonClient.getBloomFilter("bloom:" + name);
    }
}
