package com.pei.dehaze.sdk.model.order;

import lombok.Data;

@Data
public class OrderCreateForm {
    private Long packageId;
    private Long couponId;
    private String payMethod;
}
