package com.pei.dehaze.sdk.model.pkg;

import lombok.Data;

@Data
public class PriceResult {
    private Double originalPrice;
    private Double discountAmount;
    private Double couponAmount;
    private Double payableAmount;
}
