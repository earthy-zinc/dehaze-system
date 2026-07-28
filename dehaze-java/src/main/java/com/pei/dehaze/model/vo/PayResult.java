package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "支付结果")
public class PayResult {

    @Schema(description = "订单号")
    private String orderNo;

    @Schema(description = "支付方式")
    private String payMethod;

    @Schema(description = "支付链接")
    private String payUrl;

    @Schema(description = "二维码内容")
    private String qrCode;

    @Schema(description = "是否已支付完成")
    private Boolean paid;
}
