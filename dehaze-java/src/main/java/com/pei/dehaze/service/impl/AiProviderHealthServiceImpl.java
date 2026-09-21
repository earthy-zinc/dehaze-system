package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.mapper.SysDictMapper;
import com.pei.dehaze.model.entity.SysDict;
import com.pei.dehaze.service.AiProviderHealthService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * 供应商健康与熔断实现：状态存 Redis（跨端共享），阈值取自 sys_dict（ai_provider_health）。
 *
 * <p>阈值读取沿用 python 口径：进程内缓存 60s → Redis 缓存 300s → sys_dict 回源，
 * 缺省键回落种子默认值（与 config/sql/data/sys_dict.sql 同源）。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiProviderHealthServiceImpl implements AiProviderHealthService {

    private static final String CIRCUIT_KEY = "ai:provider:%d:circuit_open";
    private static final String RECOVERY_KEY = "ai:provider:%d:circuit_recovery";
    private static final String PROBE_KEY = "ai:provider:%d:half_open_probe";
    private static final String STREAK_KEY = "ai:provider:%d:fail_streak";
    private static final String WINDOW_KEY = "ai:provider:%d:window";
    private static final String LATENCY_KEY = "ai:provider:%d:latency";
    private static final String SNAPSHOT_KEY = "ai:provider:%d:health";
    private static final String HEALTH_ENABLED_KEY = "ai:provider:%d:health_enabled";
    private static final String THRESHOLDS_KEY = "ai:provider:health:thresholds";

    private static final String HEALTH_DICT_TYPE = "ai_provider_health";
    private static final long THRESHOLDS_REDIS_TTL = 300L;
    private static final long THRESHOLDS_PROCESS_TTL_MS = 60_000L;
    private static final int LATENCY_WINDOW = 500;
    private static final long BUCKET_SECONDS = 3600L;
    private static final int WINDOW_BUCKETS = 24;
    private static final long WINDOW_TTL = BUCKET_SECONDS * (WINDOW_BUCKETS + 1);

    private static final Map<String, Object> SEED_THRESHOLDS = Map.of(
            "error_rate_warn", 0.10,
            "error_rate_open", 0.30,
            "min_window_calls", 20,
            "consecutive_failures", 5,
            "circuit_cooldown", 60);

    private final StringRedisTemplate redis;
    private final SysDictMapper dictMapper;
    private final ObjectMapper objectMapper;

    private volatile long thresholdsExpireAt = 0L;
    private volatile Map<String, Object> thresholdsCache = Map.of();

    @Override
    public String getStatus(Long providerId) {
        if (!isHealthCheckEnabled(providerId)) {
            return "healthy";
        }
        if (Boolean.TRUE.equals(redis.hasKey(circuitKey(providerId)))) {
            return "open";
        }
        if (!Boolean.TRUE.equals(redis.hasKey(recoveryKey(providerId)))) {
            Map<String, Object> snapshot = readJson(snapshotKey(providerId));
            Object status = snapshot == null ? null : snapshot.get("status");
            if ("healthy".equals(status) || "suspicious".equals(status)) {
                return String.valueOf(status);
            }
            return "healthy";
        }
        // 冷却已过、恢复周期内：抢占探测租约放行单个请求（未抢到继续阻断）
        int cooldown = intOf(loadThresholds().get("circuit_cooldown"), 60);
        Boolean acquired = redis.opsForValue()
                .setIfAbsent(probeKey(providerId), "1", cooldown, TimeUnit.SECONDS);
        return Boolean.TRUE.equals(acquired) ? "half_open" : "open";
    }

    @Override
    public Map<String, Object> getSnapshot(Long providerId) {
        Map<String, Object> cached = readJson(snapshotKey(providerId));
        if (cached != null) {
            return cached;
        }
        Map<String, Object> thresholds = loadThresholds();
        long[] counts = windowCounts(providerId, true);
        long total = counts[0];
        long failed = counts[1];
        long limit = counts[2];
        List<String> rawLatency = redis.opsForList().range(latencyKey(providerId), 0, -1);
        List<Integer> latency = new ArrayList<>();
        if (rawLatency != null) {
            for (String raw : rawLatency) {
                try {
                    latency.add(Integer.parseInt(raw));
                } catch (NumberFormatException ignored) {
                    // 非数值成员视为脏数据跳过，不影响 P95 聚合
                }
            }
        }
        int p95 = p95(latency);
        boolean circuitOpen = Boolean.TRUE.equals(redis.hasKey(circuitKey(providerId)));
        String status;
        if (!isHealthCheckEnabled(providerId)) {
            status = "healthy";
        } else if (circuitOpen) {
            status = "open";
        } else if (total >= intOf(thresholds.get("min_window_calls"), 20)) {
            double errorRate = (double) failed / total;
            if (errorRate >= doubleOf(thresholds.get("error_rate_open"), 0.30)) {
                status = "open";
            } else if (errorRate >= doubleOf(thresholds.get("error_rate_warn"), 0.10)) {
                status = "suspicious";
            } else {
                status = "healthy";
            }
        } else {
            status = "healthy";
        }

        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("status", status);
        snapshot.put("circuit_open", circuitOpen);
        snapshot.put("total_calls_24h", total);
        snapshot.put("success_rate", total > 0 ? round4((double) (total - failed) / total) : 1.0);
        snapshot.put("error_rate", total > 0 ? round4((double) failed / total) : 0.0);
        snapshot.put("limit_rate", total > 0 ? round4((double) limit / total) : 0.0);
        snapshot.put("p95_latency_ms", p95);
        writeJson(snapshotKey(providerId), snapshot, 60L);
        return snapshot;
    }

    @Override
    public void setHealthCheckEnabled(Long providerId, boolean enabled) {
        redis.opsForValue().set(healthEnabledKey(providerId), enabled ? "1" : "0");
    }

    @Override
    public void clearProviderHealth(Long providerId) {
        redis.delete(List.of(
                circuitKey(providerId),
                recoveryKey(providerId),
                probeKey(providerId),
                streakKey(providerId),
                windowKey(providerId),
                latencyKey(providerId),
                snapshotKey(providerId),
                healthEnabledKey(providerId)));
    }

    @Override
    public void closeCircuit(Long providerId) {
        redis.delete(List.of(
                circuitKey(providerId),
                recoveryKey(providerId),
                probeKey(providerId),
                streakKey(providerId)));
        redis.delete(snapshotKey(providerId));
    }

    private boolean isHealthCheckEnabled(Long providerId) {
        String value = redis.opsForValue().get(healthEnabledKey(providerId));
        return value == null || !"0".equals(value);
    }

    /** 统计近 24h 调用总数/失败数/限流数（小时分桶 HMGET，开销与调用量无关） */
    private long[] windowCounts(Long providerId, boolean withLimit) {
        long bucket = Instant.now().getEpochSecond() / BUCKET_SECONDS;
        List<Object> fields = new ArrayList<>(WINDOW_BUCKETS * 3);
        for (int offset = 0; offset < WINDOW_BUCKETS; offset++) {
            for (String suffix : List.of("t", "f", "l")) {
                fields.add((bucket - offset) + ":" + suffix);
            }
        }
        List<Object> raw = redis.opsForHash().multiGet(windowKey(providerId), fields);
        long total = 0;
        long failed = 0;
        long limit = 0;
        for (int i = 0; i < WINDOW_BUCKETS; i++) {
            total += longOf(valueAt(raw, i * 3));
            failed += longOf(valueAt(raw, i * 3 + 1));
            limit += longOf(valueAt(raw, i * 3 + 2));
        }
        return withLimit ? new long[]{total, failed, limit} : new long[]{total, failed, 0};
    }

    private static Object valueAt(List<Object> raw, int index) {
        return raw == null || index >= raw.size() ? null : raw.get(index);
    }

    /**
     * 阈值：进程内 60s → Redis 300s → sys_dict 回源，缺省键回落种子默认值。
     */

    private Map<String, Object> loadThresholds() {
        long now = System.currentTimeMillis();
        if (now < thresholdsExpireAt) {
            return thresholdsCache;
        }
        Map<String, Object> data = readJson(THRESHOLDS_KEY);
        if (data == null) {
            data = new LinkedHashMap<>(SEED_THRESHOLDS);
            try {
                List<SysDict> items = dictMapper.selectList(new LambdaQueryWrapper<SysDict>()
                        .eq(SysDict::getTypeCode, HEALTH_DICT_TYPE)
                        .eq(SysDict::getStatus, 1)
                        .select(SysDict::getName, SysDict::getValue));
                for (SysDict item : items) {
                    if (CharSequenceUtil.isNotBlank(item.getName())) {
                        data.put(item.getName(), coerceScalar(item.getValue()));
                    }
                }
            } catch (Exception e) {
                log.warn("读取供应商健康阈值失败，使用种子默认: {}", e.getMessage());
            }
            writeJson(THRESHOLDS_KEY, data, THRESHOLDS_REDIS_TTL);
        }
        thresholdsCache = data;
        thresholdsExpireAt = now + THRESHOLDS_PROCESS_TTL_MS;
        return data;
    }

    private static Object coerceScalar(String raw) {
        if (raw == null) {
            return "";
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException ignored) {
            try {
                return Double.parseDouble(raw.trim());
            } catch (NumberFormatException ignoredToo) {
                return raw;
            }
        }
    }

    private static int p95(List<Integer> values) {
        if (values.isEmpty()) {
            return 0;
        }
        List<Integer> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int idx = Math.max(0, (int) (sorted.size() * 0.95) - 1);
        return sorted.get(idx);
    }

    private static double round4(double value) {
        return Math.round(value * 10000d) / 10000d;
    }

    private Map<String, Object> readJson(String key) {
        String raw = redis.opsForValue().get(key);
        if (CharSequenceUtil.isBlank(raw)) {
            return null;
        }
        try {
            return objectMapper.readValue(raw, new TypeReference<Map<String, Object>>() {
            });
        } catch (Exception e) {
            log.warn("供应商健康缓存[{}]解析失败，删除后回源: {}", key, e.getMessage());
            redis.delete(key);
            return null;
        }
    }

    private void writeJson(String key, Map<String, Object> value, long ttlSeconds) {
        try {
            redis.opsForValue().set(key, objectMapper.writeValueAsString(value), ttlSeconds, TimeUnit.SECONDS);
        } catch (Exception e) {
            log.warn("供应商健康缓存[{}]写入失败: {}", key, e.getMessage());
        }
    }

    private static int intOf(Object value, int defaultValue) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        try {
            return Integer.parseInt(String.valueOf(value));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private static long longOf(Object value) {
        if (value == null) {
            return 0L;
        }
        if (value instanceof Number number) {
            return number.longValue();
        }
        try {
            return Long.parseLong(String.valueOf(value));
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    private static double doubleOf(Object value, double defaultValue) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        try {
            return Double.parseDouble(String.valueOf(value));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private static String circuitKey(Long id) {
        return String.format(CIRCUIT_KEY, id);
    }

    private static String recoveryKey(Long id) {
        return String.format(RECOVERY_KEY, id);
    }

    private static String probeKey(Long id) {
        return String.format(PROBE_KEY, id);
    }

    private static String streakKey(Long id) {
        return String.format(STREAK_KEY, id);
    }

    private static String windowKey(Long id) {
        return String.format(WINDOW_KEY, id);
    }

    private static String latencyKey(Long id) {
        return String.format(LATENCY_KEY, id);
    }

    private static String snapshotKey(Long id) {
        return String.format(SNAPSHOT_KEY, id);
    }

    private static String healthEnabledKey(Long id) {
        return String.format(HEALTH_ENABLED_KEY, id);
    }
}
