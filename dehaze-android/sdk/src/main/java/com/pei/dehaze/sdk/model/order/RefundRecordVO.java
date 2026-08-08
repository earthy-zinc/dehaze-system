package com.pei.dehaze.sdk.model.order;

import lombok.Data;

@Data
public class RefundRecordVO {
    private Long id;
    private String refundNo;
    private Long orderId;
    private String orderNo;
    private Long userId;
    private String username;
    private Double refundAmount;
    private String reason;
    private Integer usedQuota;
    private String status;
    private String channel;
    private String channelRefundNo;
    private String applyTime;
    private String auditTime;
    private Long auditorId;
    private String auditRemark;
    private String refundTime;
    private String errorMessage;
}
