package com.pei.dehaze.sdk.model.order;

import lombok.Data;

@Data
public class RefundAuditForm {
    private Boolean approved;
    private String remark;
}
