package com.pei.dehaze.service;

import java.util.Map;

/**
 * 供应商健康与熔断（Redis 运行时聚合，不落库）。
 *
 * <p>键约定与 Python 端 provider_health_service 完全一致（跨端共享同一 Redis 状态）：
 * <pre>
 * ai:provider:{id}:circuit_open      熔断标记（TTL=冷却时长）
 * ai:provider:{id}:circuit_recovery  熔断恢复周期标记（TTL=冷却×2）
 * ai:provider:{id}:half_open_probe   半开探测租约
 * ai:provider:{id}:fail_streak       连续失败计数
 * ai:provider:{id}:window            24h 调用窗口（小时分桶 HASH）
 * ai:provider:{id}:latency           延迟 LPUSH 列表
 * ai:provider:{id}:health            聚合快照缓存
 * ai:provider:{id}:health_enabled    健康检查开关
 * ai:provider:health:thresholds      熔断阈值缓存（sys_dict ai_provider_health）
 * </pre>
 */
public interface AiProviderHealthService {

    /** 快速路径健康状态：healthy / suspicious / half_open / open */
    String getStatus(Long providerId);

    /** 看板健康快照（含成功率/限流率/P95 等），读缓存 miss 时聚合回填 */
    Map<String, Object> getSnapshot(Long providerId);

    /** 供应商 CRUD 后写入健康检查开关缓存 */
    void setHealthCheckEnabled(Long providerId, boolean enabled);

    /** 删除供应商时清理全部健康相关键 */
    void clearProviderHealth(Long providerId);

    /** 管理员手动解除熔断 */
    void closeCircuit(Long providerId);
}
