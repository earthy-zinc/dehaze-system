package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
@Schema(description = "订单创建表单")
public class OrderCreateForm {

    @Schema(description = "套餐ID", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "套餐ID不能为空")
    private Long packageId;

    @Schema(description = "用户优惠券实例ID")
    private Long couponId;

    @Schema(description = "支付方式(wechat/alipay/balance/combined)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "支付方式不能为空")
    private String payMethod;
}
