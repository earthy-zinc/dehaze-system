package com.pei.clarity.sdk.model.order;

import lombok.Data;

@Data
public class AutoRenewConfigVO {
    private Long userId;
    private Long packageId;
    private String packageName;
    private String payMethod;
    private Boolean enabled;
    private String nextRenewTime;
    private Integer failCount;
    private String closeReason;
}
