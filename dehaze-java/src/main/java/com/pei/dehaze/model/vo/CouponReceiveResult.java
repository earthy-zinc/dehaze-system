package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "优惠券领取结果")
public class CouponReceiveResult {

    @Schema(description = "用户优惠券实例ID")
    private Long userCouponId;
}
