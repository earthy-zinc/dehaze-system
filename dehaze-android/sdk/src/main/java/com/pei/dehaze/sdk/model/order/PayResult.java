package com.pei.dehaze.sdk.model.order;

import lombok.Data;

@Data
public class PayResult {
    private String orderNo;
    private String payMethod;
    private String payUrl;
    private String qrCode;
    private Boolean paid;
}
