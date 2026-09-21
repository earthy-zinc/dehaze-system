package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "退款申请表单")
@Data
public class AiRefundCreateForm {

    @NotNull(message = "原计费记录不能为空")
    private Long billingId;

    @NotNull(message = "退款积分数不能为空")
    private Integer amount;

    @NotBlank(message = "退款原因不能为空")
    private String reason;
}
