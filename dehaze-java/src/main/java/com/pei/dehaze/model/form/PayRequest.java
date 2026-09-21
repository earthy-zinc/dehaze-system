package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
@Schema(description = "支付请求")
public class PayRequest {

    @Schema(description = "支付方式(wechat/alipay/balance/combined)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "支付方式不能为空")
    private String payMethod;

    @Schema(description = "组合支付时指定的第三方渠道(wechat/alipay，组合支付必填)")
    private String channel;

    @Schema(description = "组合支付时余额部分金额(分)")
    private Long balanceAmount;
}
