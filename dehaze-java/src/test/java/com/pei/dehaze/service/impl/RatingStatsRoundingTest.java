package com.pei.dehaze.service.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.mapper.SysRatingMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.vo.RatingStatsVO;
import com.pei.dehaze.mq.RabbitMQPublisher;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.MemberService;
import com.pei.dehaze.service.SysDictService;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.test.util.ReflectionTestUtils;

/**
 * 评价统计的小数位口径（三端响应契约一致：平均评分与低分率均保留 2 位小数）。
 *
 * <p>基线为 python {@code feedback_repository.get_rating_stats}：三处都用 {@code round(value, 2)}，
 * 即对 float 的二进制精确值做 half-even —— 因此 4.125 得 4.12（非 4.13）、2.675 得 2.67。
 * Java 侧若改成 {@code Math.round(x*100)/100.0}（HALF_UP）会在这些取值上偏离 Python。
 */
@DisplayName("RatingServiceImpl 评价统计 2 位小数")
@ExtendWith(MockitoExtension.class)
class RatingStatsRoundingTest {

    @Mock
    private SysRatingMapper ratingMapper;

    @Mock
    private SysPredLogMapper predLogMapper;

    @Mock
    private SysAiModelMapper aiModelMapper;

    @Mock
    private SysUserMapper userMapper;

    @Mock
    private StringRedisTemplate stringRedisTemplate;

    @Mock
    private MemberService memberService;

    @Mock
    private SysDictService sysDictService;

    @Mock
    private ObjectProvider<RabbitMQPublisher> rabbitMQPublisherProvider;

    @Mock
    private FileService fileService;

    private final ObjectMapper objectMapper = new ObjectMapper();

    private RatingServiceImpl service;

    @BeforeEach
    void setUp() {
        service = new RatingServiceImpl(
                predLogMapper,
                aiModelMapper,
                userMapper,
                objectMapper,
                stringRedisTemplate,
                memberService,
                sysDictService,
                rabbitMQPublisherProvider,
                List.of(fileService));
        ReflectionTestUtils.setField(service, "baseMapper", ratingMapper);
    }

    @Test
    @DisplayName("全局平均评分：half-even 保留 2 位（33/8=4.125 → 4.12，与 python round 一致）")
    void globalAverageRating_halfEvenTwoDecimals() {
        when(ratingMapper.selectRatingDistribution(any(), any())).thenReturn(List.of(ratingRow(5, 1), ratingRow(4, 7)));
        when(ratingMapper.selectAllTags(any(), any())).thenReturn(List.of());
        when(ratingMapper.selectModelStats(any(), any())).thenReturn(List.of());

        RatingStatsVO stats = service.getRatingStats(LocalDateTime.now().minusDays(1), LocalDateTime.now());

        assertThat(stats.getTotalRatings()).isEqualTo(8L);
        assertThat(stats.getAverageRating()).isEqualTo(4.12);
    }

    @Test
    @DisplayName("全局平均评分：非整分的 2 位截断（13/3=4.333… → 4.33）与 python float 口径（107/40=2.675 → 2.67）")
    void globalAverageRating_threeDecimals() {
        when(ratingMapper.selectRatingDistribution(any(), any()))
                .thenReturn(List.of(ratingRow(3, 27), ratingRow(2, 13)));
        when(ratingMapper.selectAllTags(any(), any())).thenReturn(List.of());
        when(ratingMapper.selectModelStats(any(), any())).thenReturn(List.of());

        RatingStatsVO stats = service.getRatingStats(LocalDateTime.now().minusDays(1), LocalDateTime.now());

        // 107/40 的 double 为 2.6749999999999998，python round(...,2) 得 2.67（非四舍五入的 2.68）
        assertThat(stats.getTotalRatings()).isEqualTo(40L);
        assertThat(stats.getAverageRating()).isEqualTo(2.67);
    }

    @Test
    @DisplayName("模型维度：averageRating 2 位小数，lowRatingRate 按百分比 2 位小数")
    void modelStats_twoDecimals() {
        when(ratingMapper.selectRatingDistribution(any(), any())).thenReturn(List.of());
        when(ratingMapper.selectAllTags(any(), any())).thenReturn(List.of());
        when(ratingMapper.selectModelStats(any(), any()))
                .thenReturn(List.of(
                        modelRow("aecrnet", "AECRNet", new BigDecimal("2.6750"), 40, 1),
                        modelRow("dcp", "DCP", new BigDecimal("4.1250"), 32, 1),
                        modelRow("ffa", "FFA", new BigDecimal("4.6667"), 3, 3)));

        RatingStatsVO stats = service.getRatingStats(LocalDateTime.now().minusDays(1), LocalDateTime.now());

        List<RatingStatsVO.ModelStat> modelStats = stats.getModelStats();
        assertThat(modelStats).hasSize(3);

        assertThat(modelStats.get(0).getAverageRating()).isEqualTo(2.67);
        assertThat(modelStats.get(0).getLowRatingRate()).isEqualTo(2.5);

        assertThat(modelStats.get(1).getAverageRating()).isEqualTo(4.12);
        assertThat(modelStats.get(1).getLowRatingRate()).isEqualTo(3.12);

        assertThat(modelStats.get(2).getAverageRating()).isEqualTo(4.67);
        assertThat(modelStats.get(2).getLowRatingRate()).isEqualTo(100.0);
    }

    @Test
    @DisplayName("边界：无评价时低分率为 0、平均分为 0，集合非 null 且为空")
    void emptyStatsAreZeroAndNonNull() {
        when(ratingMapper.selectRatingDistribution(any(), any())).thenReturn(List.of());
        when(ratingMapper.selectAllTags(any(), any())).thenReturn(List.of());
        when(ratingMapper.selectModelStats(any(), any())).thenReturn(List.of());

        RatingStatsVO stats = service.getRatingStats(LocalDateTime.now().minusDays(1), LocalDateTime.now());

        assertThat(stats.getTotalRatings()).isZero();
        assertThat(stats.getAverageRating()).isZero();
        assertThat(stats.getRatingDistribution()).hasSize(5).containsValues(0L);
        assertThat(stats.getModelStats()).isNotNull().isEmpty();
        assertThat(stats.getPositiveTagRanking()).isNotNull().isEmpty();
        assertThat(stats.getNegativeTagRanking()).isNotNull().isEmpty();
    }

    @Test
    @DisplayName("边界：分组内评价数为 0 时低分率为 0（不产生 NaN）")
    void zeroTotalLowRatingRateIsZero() {
        when(ratingMapper.selectRatingDistribution(any(), any())).thenReturn(List.of());
        when(ratingMapper.selectAllTags(any(), any())).thenReturn(List.of());
        when(ratingMapper.selectModelStats(any(), any()))
                .thenReturn(List.of(modelRow("aecrnet", "AECRNet", new BigDecimal("4.0000"), 0, 0)));

        RatingStatsVO stats = service.getRatingStats(LocalDateTime.now().minusDays(1), LocalDateTime.now());

        assertThat(stats.getModelStats().get(0).getLowRatingRate()).isZero();
    }

    private static Map<String, Object> ratingRow(int rating, long count) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("rating", rating);
        row.put("cnt", count);
        return row;
    }

    private static Map<String, Object> modelRow(
            String modelId, String modelName, BigDecimal avgRating, long total, long lowCount) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("modelId", modelId);
        row.put("modelName", modelName);
        row.put("avgRating", avgRating);
        row.put("total", total);
        row.put("lowCount", lowCount);
        return row;
    }
}
