package com.pei.dehaze.sdk.model.member;

import lombok.Data;

@Data
public class GrowthLogVO {
    private Long id;
    private String changeType;
    private Integer changeValue;
    private Integer balance;
    private String relatedId;
    private String reason;
    private Long operatorId;
    private String createTime;
}
