package com.pei.dehaze.sdk.model.order;

import lombok.Data;

@Data
public class AutoRenewConfigForm {
    private Long packageId;
    private String payMethod;
    private Boolean enabled;
}
