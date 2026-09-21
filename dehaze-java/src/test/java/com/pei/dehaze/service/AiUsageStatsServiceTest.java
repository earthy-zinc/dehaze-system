package com.pei.dehaze.service;

import com.pei.dehaze.mapper.AiInsightMapper;
import com.pei.dehaze.model.read.DowngradeRead;
import com.pei.dehaze.model.read.ModelUsageRead;
import com.pei.dehaze.model.read.ProviderRead;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.HashOperations;
import org.springframework.data.redis.core.ListOperations;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

/**
 * AI 运营统计服务单测：供应商健康快照读缓存 / 缺失时窗口聚合回填（三态判定）、模型用量别名回填、降级与 Key 故障。
 *
 * <p>健康快照口径与 python 一致（24 小时分桶窗口 + 阈值来自 sys_dict/Redis）；
 * 聚合错位会误报 healthy/open，直接影响运维对故障的处置判断。
 */
@DisplayName("AiUsageStatsService 运营统计")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiUsageStatsServiceTest {

    private static final Long PROVIDER_ID = 1L;

    @Mock
    private AiInsightMapper insightMapper;
    @Mock
    private StringRedisTemplate stringRedisTemplate;
    @Mock
    private ValueOperations<String, String> valueOperations;
    @Mock
    private HashOperations<String, Object, Object> hashOperations;
    @Mock
    private ListOperations<String, String> listOperations;

    @InjectMocks
    private AiUsageStatsService service;

    @BeforeEach
    void setUp() {
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(stringRedisTemplate.opsForHash()).thenReturn(hashOperations);
        when(stringRedisTemplate.opsForList()).thenReturn(listOperations);
    }

    private ProviderRead provider() {
        ProviderRead provider = new ProviderRead();
        provider.setId(PROVIDER_ID);
        provider.setDisplayName("主供应商");
        return provider;
    }

    /** 24 桶 × (t/f/l) = 72 个窗口字段，仅首桶注入指标 */
    private List<Object> window(long total, long failed, long limit) {
        List<Object> values = new ArrayList<>(72);
        for (int i = 0; i < 72; i++) {
            values.add("0");
        }
        values.set(0, String.valueOf(total));
        values.set(1, String.valueOf(failed));
        values.set(2, String.valueOf(limit));
        return values;
    }

    private void stubWindow(long total, long failed, long limit) {
        when(stringRedisTemplate.opsForValue().get("ai:provider:" + PROVIDER_ID + ":health")).thenReturn(null);
        when(stringRedisTemplate.opsForValue().get("ai:provider:health:thresholds")).thenReturn(null);
        when(hashOperations.multiGet(anyString(), anyList())).thenReturn(window(total, failed, limit));
        when(listOperations.range(anyString(), eq(0L), eq(-1L))).thenReturn(List.of("10", "20", "30"));
        when(stringRedisTemplate.hasKey(anyString())).thenReturn(false);
        when(stringRedisTemplate.opsForValue().get("ai:provider:" + PROVIDER_ID + ":health_enabled")).thenReturn(null);
    }

    @Test
    @DisplayName("统计响应含三段结构：供应商健康 / 模型用量 / 降级与故障")
    void usageStatsHasThreeSections() {
        when(insightMapper.listProviders()).thenReturn(List.of());
        when(insightMapper.listModelUsage(any(), any())).thenReturn(List.of());
        when(insightMapper.listDowngradeByModel(any(), any())).thenReturn(List.of());
        when(stringRedisTemplate.keys(anyString())).thenReturn(Set.of());

        var vo = service.getUsageStats(LocalDateTime.now().minusDays(1), LocalDateTime.now());

        assertThat(vo.getProviderHealth()).isEmpty();
        assertThat(vo.getModelUsage()).isEmpty();
        assertThat(vo.getDegradeFault().getDowngradeFrequency()).isEmpty();
        assertThat(vo.getDegradeFault().getKeyFailoverCount()).isZero();
    }

    @Test
    @DisplayName("供应商健康：命中运行面快照时直接读取，不做窗口聚合")
    void providerHealthReadsCachedSnapshot() {
        when(insightMapper.listProviders()).thenReturn(List.of(provider()));
        when(stringRedisTemplate.opsForValue().get("ai:provider:" + PROVIDER_ID + ":health"))
                .thenReturn("{\"status\":\"open\",\"circuit_open\":true,\"total_calls_24h\":120,"
                        + "\"success_rate\":0.9,\"limit_rate\":0.25,\"p95_latency_ms\":800}");

        var health = service.getUsageStats(null, null).getProviderHealth().get(0);

        assertThat(health.getProviderId()).isEqualTo(PROVIDER_ID);
        assertThat(health.getProviderName()).isEqualTo("主供应商");
        assertThat(health.getHealth()).isEqualTo("open");
        assertThat(health.getCallCount()).isEqualTo(120);
        assertThat(health.getCircuitOpen()).isTrue();
        assertThat(health.getP95LatencyMs()).isEqualTo(800);
        assertThat(health.getRate429()).isEqualTo(0.25);
    }

    @Test
    @DisplayName("供应商健康：快照缺失时按 24 小时窗口聚合，错误率超开闸阈值判 open 并回填缓存")
    void providerHealthAggregatesWindowWhenSnapshotMissing() {
        when(insightMapper.listProviders()).thenReturn(List.of(provider()));
        stubWindow(100, 35, 7);

        var health = service.getUsageStats(null, null).getProviderHealth().get(0);

        assertThat(health.getHealth()).isEqualTo("open");
        assertThat(health.getCallCount()).isEqualTo(100);
        assertThat(health.getSuccessRate()).isEqualTo(0.65);
        assertThat(health.getRate429()).isEqualTo(0.07);
        assertThat(health.getP95LatencyMs()).isEqualTo(20);
        verifySnapshotBackfilled();
    }

    @Test
    @DisplayName("供应商健康：错误率处于预警区间判 suspicious，样本不足最小窗口时判 healthy")
    void providerHealthSuspiciousAndHealthyBranches() {
        when(insightMapper.listProviders()).thenReturn(List.of(provider()));
        stubWindow(100, 15, 0);

        assertThat(service.getUsageStats(null, null).getProviderHealth().get(0).getHealth())
                .isEqualTo("suspicious");

        stubWindow(10, 9, 0);
        assertThat(service.getUsageStats(null, null).getProviderHealth().get(0).getHealth())
                .isEqualTo("healthy");
    }

    @Test
    @DisplayName("模型用量：有别名用别名，无别名回退模型 ID，并透传 Token/积分")
    void modelUsageFillsDisplayName() {
        ModelUsageRead aliased = new ModelUsageRead();
        aliased.setModelId("qwen3-0.6b");
        aliased.setCallCount(12L);
        aliased.setInputTokens(1000L);
        aliased.setOutputTokens(500L);
        aliased.setCredits(30L);
        ModelUsageRead plain = new ModelUsageRead();
        plain.setModelId("unknown-model");
        plain.setCallCount(1L);
        when(insightMapper.listProviders()).thenReturn(List.of());
        when(insightMapper.listModelUsage(any(), any())).thenReturn(List.of(aliased, plain));
        when(insightMapper.listModelDisplayNames(List.of("qwen3-0.6b", "unknown-model")))
                .thenReturn(List.of(Map.of("modelId", "qwen3-0.6b", "name", "Qwen3 0.6B")));
        when(insightMapper.listDowngradeByModel(any(), any())).thenReturn(List.of());
        when(stringRedisTemplate.keys(anyString())).thenReturn(Set.of());

        var usage = service.getUsageStats(null, null).getModelUsage();

        assertThat(usage.get(0).getDisplayName()).isEqualTo("Qwen3 0.6B");
        assertThat(usage.get(0).getCredits()).isEqualTo(30L);
        assertThat(usage.get(1).getDisplayName()).isEqualTo("unknown-model");
    }

    @Test
    @DisplayName("降级与故障：零次降级不展示，Key 故障数取冷却中 Key 快照")
    void degradeFaultFiltersZeroAndCountsUnavailableKeys() {
        DowngradeRead hit = new DowngradeRead();
        hit.setModelId("qwen3-0.6b");
        hit.setCnt(3L);
        DowngradeRead zero = new DowngradeRead();
        zero.setModelId("other");
        zero.setCnt(0L);
        when(insightMapper.listProviders()).thenReturn(List.of());
        when(insightMapper.listModelUsage(any(), any())).thenReturn(List.of());
        when(insightMapper.listDowngradeByModel(any(), any())).thenReturn(List.of(hit, zero));
        when(stringRedisTemplate.keys("ai:provider_key:*:unavailable"))
                .thenReturn(Set.of("ai:provider_key:1:unavailable", "ai:provider_key:2:unavailable"));

        var fault = service.getUsageStats(null, null).getDegradeFault();

        assertThat(fault.getDowngradeFrequency()).hasSize(1);
        assertThat(fault.getDowngradeFrequency().get(0).getModelId()).isEqualTo("qwen3-0.6b");
        assertThat(fault.getKeyFailoverCount()).isEqualTo(2);
    }

    private void verifySnapshotBackfilled() {
        org.mockito.Mockito.verify(valueOperations).set(
                eq("ai:provider:" + PROVIDER_ID + ":health"), anyString(), any(java.time.Duration.class));
    }
}
