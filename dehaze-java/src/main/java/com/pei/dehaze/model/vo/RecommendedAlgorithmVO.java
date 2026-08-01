package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "推荐算法项")
public class RecommendedAlgorithmVO {

    @Schema(description = "推荐记录ID（sys_recommendation.id）")
    private Long recommendationId;

    @Schema(description = "算法ID")
    private Long algorithmId;

    @Schema(description = "算法名称")
    private String algorithmName;

    @Schema(description = "匹配度(0-100)")
    private Integer matchScore;

    @Schema(description = "推荐理由")
    private String reason;

    @Schema(description = "算法评分(0-5)")
    private Double rating;

    @Schema(description = "预估处理耗时(ms)")
    private Integer estimatedTime;

    @Schema(description = "预期效果描述")
    private String effectDescription;
}
