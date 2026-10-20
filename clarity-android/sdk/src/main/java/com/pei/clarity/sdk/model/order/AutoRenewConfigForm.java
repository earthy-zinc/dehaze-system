package com.pei.clarity.sdk.model.order;

import lombok.Data;

@Data
public class AutoRenewConfigForm {
    private Long packageId;
    private String payMethod;
    private Boolean enabled;
}
