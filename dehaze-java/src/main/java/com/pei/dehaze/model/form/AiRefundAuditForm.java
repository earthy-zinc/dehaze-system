package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "退款审核表单")
@Data
public class AiRefundAuditForm {

    @NotNull(message = "审核结论不能为空")
    private Boolean approved;

    private String auditRemark;
}
