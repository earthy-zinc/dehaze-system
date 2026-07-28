package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
@Schema(description = "反馈分配表单")
public class FeedbackAssignForm {

    @Schema(description = "处理人ID", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "处理人ID不能为空")
    private Long assigneeId;
}
