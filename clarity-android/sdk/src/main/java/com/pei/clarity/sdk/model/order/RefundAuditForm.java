package com.pei.clarity.sdk.model.order;

import lombok.Data;

@Data
public class RefundAuditForm {
    private Boolean approved;
    private String remark;
}
