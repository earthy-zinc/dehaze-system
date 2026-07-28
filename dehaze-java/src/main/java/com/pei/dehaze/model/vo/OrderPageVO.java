package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "订单后台分页VO")
public class OrderPageVO extends MyOrderVO {

    @Schema(description = "用户ID")
    private Long userId;

    @Schema(description = "用户名")
    private String username;

    @Schema(description = "原价（分）")
    private Long originalPrice;

    @Schema(description = "促销折扣（分）")
    private Long discountAmount;

    @Schema(description = "优惠券抵扣（分）")
    private Long couponAmount;
}
