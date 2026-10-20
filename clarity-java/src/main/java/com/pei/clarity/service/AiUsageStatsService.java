package com.pei.clarity.service;

import cn.hutool.json.JSONConfig;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.pei.clarity.mapper.AiInsightMapper;
import com.pei.clarity.model.read.DowngradeRead;
import com.pei.clarity.model.read.ModelUsageRead;
import com.pei.clarity.model.read.ProviderRead;
import com.pei.clarity.model.vo.AiDegradeFaultVO;
import com.pei.clarity.model.vo.AiDowngradeVO;
import com.pei.clarity.model.vo.AiModelUsageVO;
import com.pei.clarity.model.vo.AiProviderHealthVO;
import com.pei.clarity.model.vo.AiUsageStatsVO;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

/**
 * AI 供应商/模型运营统计服务（GET /api/v1/ai/usage/stats）。
 *
 * <p>数据口径对齐 clarity-python {@code ai_usage_stats_service}：供应商健康读 Redis 快照
 * （{@code ai:provider:{id}:health}，缺失时按小时分桶窗口聚合回填）、模型用量按 sys_ai_billing
 * 实际模型聚合、降级频次按 actual_model 非空聚合、Key 故障数为冷却中 Key 数。
 *
 * @author clarity
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiUsageStatsService {

    private static final String HEALTH_SNAPSHOT_KEY = "ai:provider:%s:health";

    private static final String HEALTH_ENABLED_KEY = "ai:provider:%s:health_enabled";

    private static final String CIRCUIT_KEY = "ai:provider:%s:circuit_open";

    private static final String WINDOW_KEY = "ai:provider:%s:window";

    private static final String LATENCY_KEY = "ai:provider:%s:latency";

    private static final String THRESHOLDS_KEY = "ai:provider:health:thresholds";

    private static final String KEY_UNAVAILABLE_PATTERN = "ai:provider_key:*:unavailable";

    private static final int WINDOW_BUCKETS = 24;

    private static final long BUCKET_SECONDS = 3600L;

    private static final Duration SNAPSHOT_TTL = Duration.ofSeconds(60);

    private final AiInsightMapper insightMapper;

    private final StringRedisTemplate stringRedisTemplate;

    public AiUsageStatsVO getUsageStats(LocalDateTime startTime, LocalDateTime endTime) {
        AiUsageStatsVO vo = new AiUsageStatsVO();
        vo.setProviderHealth(providerHealth());
        vo.setModelUsage(modelUsage(startTime, endTime));
        vo.setDegradeFault(degradeFault(startTime, endTime));
        return vo;
    }

    private List<AiProviderHealthVO> providerHealth() {
        List<AiProviderHealthVO> items = new ArrayList<>();
        for (ProviderRead provider : insightMapper.listProviders()) {
            Map<String, Object> snapshot = healthSnapshot(provider.getId());
            AiProviderHealthVO vo = new AiProviderHealthVO();
            vo.setProviderId(provider.getId());
            vo.setProviderName(provider.getDisplayName());
            vo.setHealth(String.valueOf(snapshot.get("status")));
            vo.setCallCount(intValue(snapshot.get("total_calls_24h")));
            // 快照里比率/P95 为 null 表示近 24h 无调用（无样本），保持 null 交给前端显示「—」
            vo.setSuccessRate(nullableDouble(snapshot.get("success_rate")));
            vo.setRate429(nullableDouble(snapshot.get("limit_rate")));
            vo.setP95LatencyMs(nullableInt(snapshot.get("p95_latency_ms")));
            vo.setCircuitOpen(Boolean.TRUE.equals(snapshot.get("circuit_open")));
            items.add(vo);
        }
        return items;
    }

    /**
     * 供应商健康快照：优先读运行面回填的缓存，缺失时按 24 小时分桶窗口聚合后回填（口径与 python 一致）
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> healthSnapshot(Long providerId) {
        String cached = stringRedisTemplate.opsForValue().get(String.format(HEALTH_SNAPSHOT_KEY, providerId));
        if (cached != null) {
            return JSONUtil.parseObj(cached);
        }
        double errorRateWarn = 0.10;
        double errorRateOpen = 0.30;
        int minWindowCalls = 20;
        String thresholds = stringRedisTemplate.opsForValue().get(THRESHOLDS_KEY);
        if (thresholds != null) {
            Map<String, Object> parsed = JSONUtil.parseObj(thresholds);
            errorRateWarn = doubleValue(parsed.get("error_rate_warn"));
            errorRateOpen = doubleValue(parsed.get("error_rate_open"));
            minWindowCalls = intValue(parsed.get("min_window_calls"));
        }
        long bucket = System.currentTimeMillis() / 1000 / BUCKET_SECONDS;
        List<Object> fields = new ArrayList<>();
        for (int offset = 0; offset < WINDOW_BUCKETS; offset++) {
            for (String suffix : List.of("t", "f", "l")) {
                fields.add((bucket - offset) + ":" + suffix);
            }
        }
        List<Object> raw = stringRedisTemplate.opsForHash().multiGet(String.format(WINDOW_KEY, providerId), fields);
        long total = 0;
        long failed = 0;
        long limit = 0;
        for (int i = 0; i < WINDOW_BUCKETS; i++) {
            total += parseLong(raw, i * 3);
            failed += parseLong(raw, i * 3 + 1);
            limit += parseLong(raw, i * 3 + 2);
        }
        List<String> latencyRaw = stringRedisTemplate.opsForList().range(String.format(LATENCY_KEY, providerId), 0, -1);
        int p95 = p95(latencyRaw);
        boolean circuitOpen = Boolean.TRUE.equals(stringRedisTemplate.hasKey(String.format(CIRCUIT_KEY, providerId)));
        String enabledFlag = stringRedisTemplate.opsForValue().get(String.format(HEALTH_ENABLED_KEY, providerId));
        boolean enabled = enabledFlag == null || !"0".equals(enabledFlag);
        String status = "healthy";
        if (enabled && circuitOpen) {
            status = "open";
        } else if (enabled && total >= minWindowCalls) {
            double errorRate = (double) failed / total;
            if (errorRate >= errorRateOpen) {
                status = "open";
            } else if (errorRate >= errorRateWarn) {
                status = "suspicious";
            }
        }
        // 窗口内无调用（total=0）时比率与延迟都没有样本：写 null，与 python get_health_snapshot 同口径。
        // 写 Redis 时关闭 hutool 的忽略 null（默认会丢键），否则按契约严格取值的端会当成字段缺失。
        Map<String, Object> snapshot = new HashMap<>();
        snapshot.put("status", status);
        snapshot.put("circuit_open", circuitOpen);
        snapshot.put("total_calls_24h", total);
        snapshot.put("success_rate", total == 0 ? null : round((double) (total - failed) / total));
        snapshot.put("limit_rate", total == 0 ? null : round((double) limit / total));
        snapshot.put("p95_latency_ms", total == 0 ? null : p95);
        stringRedisTemplate
                .opsForValue()
                .set(
                        String.format(HEALTH_SNAPSHOT_KEY, providerId),
                        JSONUtil.toJsonStr(
                                new JSONObject(snapshot, JSONConfig.create().setIgnoreNullValue(false))),
                        SNAPSHOT_TTL);
        return snapshot;
    }

    private List<AiModelUsageVO> modelUsage(LocalDateTime startTime, LocalDateTime endTime) {
        List<ModelUsageRead> rows = insightMapper.listModelUsage(startTime, endTime);
        Map<String, String> names = new HashMap<>();
        List<String> modelIds = rows.stream().map(ModelUsageRead::getModelId).toList();
        if (!modelIds.isEmpty()) {
            for (Map<String, Object> row : insightMapper.listModelDisplayNames(modelIds)) {
                names.put(String.valueOf(row.get("modelId")), String.valueOf(row.get("name")));
            }
        }
        List<AiModelUsageVO> items = new ArrayList<>();
        for (ModelUsageRead row : rows) {
            AiModelUsageVO vo = new AiModelUsageVO();
            vo.setModelId(row.getModelId());
            vo.setDisplayName(names.getOrDefault(row.getModelId(), row.getModelId()));
            vo.setCallCount(row.getCallCount());
            vo.setInputTokens(row.getInputTokens());
            vo.setOutputTokens(row.getOutputTokens());
            vo.setCredits(row.getCredits());
            items.add(vo);
        }
        return items;
    }

    private AiDegradeFaultVO degradeFault(LocalDateTime startTime, LocalDateTime endTime) {
        List<AiDowngradeVO> downgrade = new ArrayList<>();
        for (DowngradeRead row : insightMapper.listDowngradeByModel(startTime, endTime)) {
            if (row.getCnt() != null && row.getCnt() > 0) {
                AiDowngradeVO item = new AiDowngradeVO();
                item.setModelId(row.getModelId());
                item.setCount(row.getCnt());
                downgrade.add(item);
            }
        }
        AiDegradeFaultVO vo = new AiDegradeFaultVO();
        vo.setDowngradeFrequency(downgrade);
        vo.setKeyFailoverCount(countUnavailableKeys());
        return vo;
    }

    /**
     * 处于冷却期（临时不可用）的 Key 数：近期失败切换的当前快照
     */
    private int countUnavailableKeys() {
        var keys = stringRedisTemplate.keys(KEY_UNAVAILABLE_PATTERN);
        return keys == null ? 0 : keys.size();
    }

    private long parseLong(List<Object> values, int index) {
        Object value = values == null || index >= values.size() ? null : values.get(index);
        if (value == null) {
            return 0;
        }
        try {
            return Long.parseLong(String.valueOf(value));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private int p95(List<String> latencyRaw) {
        if (latencyRaw == null || latencyRaw.isEmpty()) {
            return 0;
        }
        List<Integer> values = new ArrayList<>();
        for (String raw : latencyRaw) {
            try {
                values.add(Integer.parseInt(raw));
            } catch (NumberFormatException ignored) {
                // 非数值延迟样本直接跳过
            }
        }
        if (values.isEmpty()) {
            return 0;
        }
        values.sort(Integer::compareTo);
        int index = Math.max(0, (int) (values.size() * 0.95) - 1);
        return values.get(index);
    }

    private int intValue(Object value) {
        return value instanceof Number number ? number.intValue() : 0;
    }

    private double doubleValue(Object value) {
        return value instanceof Number number ? number.doubleValue() : 0.0;
    }

    /** 快照缺键或为 null 表示无样本，保持 null（不可落成 0，否则前端显示成 100% 成功、0ms） */
    private Double nullableDouble(Object value) {
        return value instanceof Number number ? number.doubleValue() : null;
    }

    private Integer nullableInt(Object value) {
        return value instanceof Number number ? number.intValue() : null;
    }

    private double round(double value) {
        return new java.math.BigDecimal(value)
                .setScale(4, java.math.RoundingMode.HALF_UP)
                .doubleValue();
    }
}
