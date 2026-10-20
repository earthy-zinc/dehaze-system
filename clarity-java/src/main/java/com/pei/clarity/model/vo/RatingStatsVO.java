package com.pei.clarity.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;
import java.util.Map;
import lombok.Data;

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

    @Schema(description = "模型维度统计")
    private List<ModelStat> modelStats;

    @Data
    @Schema(description = "标签统计项")
    public static class TagCount {
        private String tag;
        private Long count;
    }

    @Data
    @Schema(description = "模型统计项")
    public static class ModelStat {
        @Schema(description = "图像模型标识")
        private String modelId;

        @Schema(description = "模型名称")
        private String modelName;

        @Schema(description = "平均评分（保留 2 位小数）")
        private Double averageRating;

        @Schema(description = "评价总数")
        private Long totalRatings;

        @Schema(description = "低分率(%)（保留 2 位小数）")
        private Double lowRatingRate;
    }
}
