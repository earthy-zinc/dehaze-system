package com.pei.clarity.sdk.model.pkg;

import lombok.Data;

@Data
public class PriceResult {
    private Long originalPrice;
    private Long discountAmount;
    private Long couponAmount;
    private Long payableAmount;
}
