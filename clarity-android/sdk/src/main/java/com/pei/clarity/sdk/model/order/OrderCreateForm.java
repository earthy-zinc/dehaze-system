package com.pei.clarity.sdk.model.order;

import lombok.Data;

@Data
public class OrderCreateForm {
    private Long packageId;
    private Long couponId;
    private String payMethod;
}
