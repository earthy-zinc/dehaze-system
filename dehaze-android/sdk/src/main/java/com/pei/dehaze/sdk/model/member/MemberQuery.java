package com.pei.dehaze.sdk.model.member;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class MemberQuery extends PageQuery {
    private String keywords;
    private String levelCode;
    private Integer status;
    private String expireTimeStart;
    private String expireTimeEnd;
    private Integer growthMin;
    private Integer growthMax;
}
