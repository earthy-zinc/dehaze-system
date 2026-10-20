package com.pei.clarity.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "推荐图像模型项")
public class RecommendedModelVO {

    @Schema(description = "推荐记录ID（sys_recommendation.id）")
    private Long recommendationId;

    @Schema(description = "图像模型标识（统一模型规范 slug）")
    private String modelId;

    @Schema(description = "模型名称")
    private String modelName;

    @Schema(description = "匹配度(0-100)")
    private Integer matchScore;

    @Schema(description = "推荐理由")
    private String reason;

    @Schema(description = "模型评分(0-5)，待真实数据采集后填充")
    private Double rating;

    @Schema(description = "预估处理耗时(ms)，待真实数据采集后填充")
    private Integer estimatedTime;

    @Schema(description = "预期效果描述")
    private String effectDescription;
}
