package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "价格计算结果")
public class PriceResult {

    @Schema(description = "原价（分）")
    private Long originalPrice;

    @Schema(description = "促销折扣（分）")
    private Long discountAmount;

    @Schema(description = "优惠券抵扣（分）")
    private Long couponAmount;

    @Schema(description = "应付金额（分）")
    private Long payableAmount;
}
