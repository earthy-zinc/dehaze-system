package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "余额退款申请表单")
public class BalanceRefundForm {

    @Schema(description = "关联订单ID(可空)")
    private Long orderId;

    @Schema(description = "退款金额(分,为空时按可用余额)")
    private Long amount;
}
