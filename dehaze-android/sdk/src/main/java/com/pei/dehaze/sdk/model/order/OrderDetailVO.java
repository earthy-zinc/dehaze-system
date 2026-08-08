package com.pei.dehaze.sdk.model.order;

import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.List;

@Data
@EqualsAndHashCode(callSuper = true)
public class OrderDetailVO extends OrderPageVO {
    private String expireTime;
    private String effectiveTime;
    private String cancelReason;
    private Integer isAutoRenew;
    private List<PaymentRecordVO> paymentRecords;
    private RefundRecordVO refundRecord;
}
