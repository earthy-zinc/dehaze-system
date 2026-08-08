package com.pei.dehaze.sdk.model.member;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class GrowthLogQuery extends PageQuery {
    private String changeType;
    private String startTime;
    private String endTime;
}
