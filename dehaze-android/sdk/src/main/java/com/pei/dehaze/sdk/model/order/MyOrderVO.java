package com.pei.dehaze.sdk.model.order;

import lombok.Data;

@Data
public class MyOrderVO {
    private Long id;
    private String orderNo;
    private String packageName;
    private String packageLevel;
    private Double payableAmount;
    private Double paidAmount;
    private String payMethod;
    private String status;
    private String createTime;
    private String paidTime;
    private String packageExpireTime;
}
