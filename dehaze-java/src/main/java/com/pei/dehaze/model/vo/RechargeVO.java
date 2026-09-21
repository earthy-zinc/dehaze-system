package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "余额充值订单VO")
public class RechargeVO {

    @Schema(description = "充值单号")
    private String rechargeNo;

    @Schema(description = "支付方式")
    private String payMethod;

    @Schema(description = "支付链接")
    private String payUrl;

    @Schema(description = "支付二维码")
    private String qrCode;
}
