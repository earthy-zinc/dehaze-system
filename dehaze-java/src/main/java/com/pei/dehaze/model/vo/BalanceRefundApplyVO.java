package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "余额退款申请结果VO")
public class BalanceRefundApplyVO {

    @Schema(description = "退款单号")
    private String refundNo;

    @Schema(description = "退款金额(分)")
    private Long amount;
}
