package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import lombok.Data;

@Data
@Schema(description = "余额充值创建表单")
public class RechargeCreateForm {

    @NotNull(message = "充值金额不能为空")
    @Positive(message = "充值金额必须大于0")
    @Schema(description = "充值金额(分)")
    private Long amount;

    @NotBlank(message = "支付方式不能为空")
    @Schema(description = "支付方式(wechat/alipay)")
    private String payMethod;
}
