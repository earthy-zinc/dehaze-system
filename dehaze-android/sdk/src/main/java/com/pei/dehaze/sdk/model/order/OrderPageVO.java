package com.pei.dehaze.sdk.model.order;

import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class OrderPageVO extends MyOrderVO {
    private Long userId;
    private String username;
    private Long originalPrice;
    private Long discountAmount;
    private Long couponAmount;
}
