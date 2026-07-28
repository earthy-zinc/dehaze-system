package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
@Schema(description = "反馈关闭表单")
public class FeedbackCloseForm {

    @Schema(description = "关闭原因", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "关闭原因不能为空")
    private String closeReason;
}
