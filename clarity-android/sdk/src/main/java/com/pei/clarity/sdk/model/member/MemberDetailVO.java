package com.pei.clarity.sdk.model.member;

import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class MemberDetailVO extends MemberProfileVO {
    private String levelSource;
    private Integer totalConsumption;
    private String becomeMemberTime;
    private String frozenReason;
    private String frozenTime;
    private Integer quotaResetMonth;
}
