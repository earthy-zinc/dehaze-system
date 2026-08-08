package com.pei.dehaze.sdk.model.pkg;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class PackageQuery extends PageQuery {
    private String name;
    private String levelCode;
    private String period;
    private Integer status;
    private String startTime;
    private String endTime;
}
