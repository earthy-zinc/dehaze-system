package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
@Schema(description = "推荐反馈表单")
public class RecommendationFeedbackForm {

    @Schema(description = "推荐记录ID")
    @NotNull(message = "推荐记录ID不能为空")
    private Long recommendationId;

    @Schema(description = "反馈：true=有用，false=无用")
    @NotNull(message = "反馈不能为空")
    private Boolean useful;
}
