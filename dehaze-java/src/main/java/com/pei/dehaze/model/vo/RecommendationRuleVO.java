package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "推荐规则")
public class RecommendationRuleVO {

    @Schema(description = "规则ID")
    private Long id;

    @Schema(description = "规则名称")
    private String ruleName;

    @Schema(description = "场景类型")
    private String sceneType;

    @Schema(description = "候选算法ID列表")
    private List<Long> algorithmIds;

    @Schema(description = "权重(0-100)")
    private Integer weight;

    @Schema(description = "是否启用")
    private Boolean enabled;
}
