package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
@Schema(description = "评价统计VO")
public class RatingStatsVO {

    @Schema(description = "评价总数")
    private Long totalRatings;

    @Schema(description = "平均评分")
    private Double averageRating;

    @Schema(description = "评分分布(1-5星各占比)")
    private Map<Integer, Long> ratingDistribution;

    @Schema(description = "正面标签排行")
    private List<TagCount> positiveTagRanking;

    @Schema(description = "负面标签排行")
    private List<TagCount> negativeTagRanking;

    @Schema(description = "算法维度统计")
    private List<AlgorithmStat> algorithmStats;

    @Data
    @Schema(description = "标签统计项")
    public static class TagCount {
        private String tag;
        private Long count;
    }

    @Data
    @Schema(description = "算法统计项")
    public static class AlgorithmStat {
        private Long algorithmId;
        private String algorithmName;
        private Double averageRating;
        private Long totalRatings;
        private Double lowRatingRate;
    }
}
