package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
@Schema(description = "退款申请表单")
public class RefundApplyForm {

    @Schema(description = "退款原因", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "退款原因不能为空")
    private String reason;

    @Schema(description = "自定义补充说明")
    private String customReason;
}
