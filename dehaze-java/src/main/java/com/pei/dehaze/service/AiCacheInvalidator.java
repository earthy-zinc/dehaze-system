package com.pei.dehaze.service;

import cn.hutool.json.JSONObject;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.UUID;

/**
 * AI 域缓存失效器。
 *
 * <p>Agent 相关缓存由运行面（dehaze-python）以裸 Redis 键 + 进程内 L1 两级缓存维护，
 * 键规范：{@code ai:agent:{agentCode}}、{@code ai:agent:{agentId}:skills|mcp|subagents|published}、
 * {@code ai:agent:list:enabled}。Java 侧写操作必须按同一键规范失效，否则跨端读到脏快照；
 * 同时发布 {@code cache:invalidation} 广播，通知其他实例（含 python）清除进程内 L1。
 *
 * @author dehaze
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class AiCacheInvalidator {

    private static final String CACHE_INVALIDATION_CHANNEL = "cache:invalidation";

    private final StringRedisTemplate stringRedisTemplate;

    private final String instanceId = UUID.randomUUID().toString();

    /**
     * Agent 更新/启停/删除/关联变更后的缓存失效（事务提交后执行）
     */
    public void evictAgentCaches(String agentCode, Long agentId) {
        evict("ai:agent:" + agentCode,
                "ai:agent:" + agentId + ":skills",
                "ai:agent:" + agentId + ":mcp",
                "ai:agent:" + agentId + ":subagents",
                "ai:agent:" + agentId + ":published",
                "ai:agent:list:enabled");
    }

    /**
     * 版本发布/回滚后的已发布版本缓存失效（事务提交后执行）
     */
    public void evictPublishedVersion(Long agentId) {
        evict("ai:agent:" + agentId + ":published");
    }

    /**
     * MCP Server/命名空间/凭据变更后的推理图缓存失效广播（事务提交后执行）。
     *
     * <p>推理图由 python 运行面构建并缓存在进程内，java 原生改 MCP（共享库）不会触发
     * python 重建图，图内装载的 MCP 外部工具集停留在旧值；python 收到本消息即失效。
     */
    public void evictReasoningGraphs() {
        AiTxSupport.afterCommit(() -> {
            try {
                JSONObject msg = new JSONObject();
                msg.set("type", "ai_graph_invalidate");
                msg.set("senderId", instanceId);
                stringRedisTemplate.convertAndSend(CACHE_INVALIDATION_CHANNEL, msg.toString());
            } catch (Exception e) {
                log.warn("推理图缓存失效广播失败（不影响业务）", e);
            }
        });
    }

    public void evict(String... keys) {
        AiTxSupport.afterCommit(() -> doEvict(keys));
    }

    private void doEvict(String[] keys) {
        try {
            stringRedisTemplate.delete(Arrays.asList(keys));
            for (String key : keys) {
                JSONObject msg = new JSONObject();
                msg.set("type", "key");
                msg.set("key", key);
                msg.set("senderId", instanceId);
                stringRedisTemplate.convertAndSend(CACHE_INVALIDATION_CHANNEL, msg.toString());
            }
        } catch (Exception e) {
            log.warn("AI 缓存失效失败（不影响业务）: keys={}", Arrays.toString(keys), e);
        }
    }
}
