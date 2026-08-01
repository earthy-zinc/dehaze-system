package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "推荐规则表单")
public class RecommendationRuleForm {

    @Schema(description = "规则名称")
    @NotBlank(message = "规则名称不能为空")
    private String ruleName;

    @Schema(description = "场景类型(urban/landscape/building/night/backlight/indoor)")
    @NotBlank(message = "场景类型不能为空")
    private String sceneType;

    @Schema(description = "候选算法ID列表")
    @NotNull(message = "算法ID列表不能为空")
    private List<Long> algorithmIds;

    @Schema(description = "规则权重(0-100)")
    @NotNull(message = "权重不能为空")
    private Integer weight;

    @Schema(description = "是否启用")
    @NotNull(message = "启用状态不能为空")
    private Boolean enabled;
}
