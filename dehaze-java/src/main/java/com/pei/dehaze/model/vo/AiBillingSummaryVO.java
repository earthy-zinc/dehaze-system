package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

/** 用户端消耗汇总（当前时段总消耗 + 趋势 + 模型分布 + 缓存节省） */
@Schema(description = "用户端消耗汇总")
@Data
public class AiBillingSummaryVO {

    private Integer totalCredits;

    private Integer inputTokens;

    private Integer outputTokens;

    private List<TrendPoint> trend;

    private List<ModelDist> modelDistribution;

    private Savings savings;

    @Schema(description = "日/月消耗趋势点")
    @Data
    public static class TrendPoint {

        private String date;

        private Integer credits;

        private Integer inputTokens;

        private Integer outputTokens;
    }

    @Schema(description = "模型消耗分布")
    @Data
    public static class ModelDist {

        private String model;

        private Integer credits;

        private Integer tokens;
    }

    @Schema(description = "缓存节省汇总")
    @Data
    public static class Savings {

        private Integer cachedInputTokens;

        private Integer creditsSaved;
    }
}
