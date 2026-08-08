package com.pei.dehaze.sdk.model.order;

import lombok.Data;

@Data
public class PaymentRecordVO {
    private Long id;
    private String paymentNo;
    private String channel;
    private Double amount;
    private Integer status;
    private String callbackTime;
    private String createTime;
}
