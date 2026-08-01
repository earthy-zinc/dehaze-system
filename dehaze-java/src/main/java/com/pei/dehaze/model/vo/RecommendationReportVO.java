package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "推荐效果报表")
public class RecommendationReportVO {

    @Schema(description = "推荐总次数")
    private Long totalRecommendations;

    @Schema(description = "采纳率(0-1)")
    private Double adoptionRate;

    @Schema(description = "满意度(0-1)")
    private Double satisfactionRate;

    @Schema(description = "覆盖率(0-1)")
    private Double coverageRate;

    @Schema(description = "冷启动成功率")
    private Double coldStartSuccessRate;

    @Schema(description = "推荐效果趋势")
    private List<TrendItem> trend;

    @Data
    @Schema(description = "趋势条目")
    public static class TrendItem {

        @Schema(description = "日期")
        private String date;

        @Schema(description = "采纳率")
        private Double adoptionRate;
    }
}
