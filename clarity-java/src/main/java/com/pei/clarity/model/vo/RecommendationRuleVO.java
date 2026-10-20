package com.pei.clarity.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;
import lombok.Data;

@Data
@Schema(description = "推荐规则")
public class RecommendationRuleVO {

    @Schema(description = "规则ID")
    private Long id;

    @Schema(description = "规则名称")
    private String ruleName;

    @Schema(description = "场景类型")
    private String sceneType;

    @Schema(description = "候选图像模型标识列表（统一模型规范 slug）")
    private List<String> modelIds;

    @Schema(description = "权重(0-100)")
    private Integer weight;

    @Schema(description = "是否启用")
    private Boolean enabled;
}
