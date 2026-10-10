package com.pei.dehaze.sdk.model.pkg;

import lombok.Data;

@Data
public class PriceResult {
    private Long originalPrice;
    private Long discountAmount;
    private Long couponAmount;
    private Long payableAmount;
}
