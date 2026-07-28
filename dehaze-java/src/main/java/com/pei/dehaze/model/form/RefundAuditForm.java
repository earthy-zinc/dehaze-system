package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
@Schema(description = "退款审核表单")
public class RefundAuditForm {

    @Schema(description = "是否通过", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "审核结果不能为空")
    private Boolean approved;

    @Schema(description = "审核备注")
    private String remark;
}
