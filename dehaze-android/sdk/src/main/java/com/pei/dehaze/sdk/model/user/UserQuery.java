package com.pei.dehaze.sdk.model.user;

import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;

/**
 * 用户查询对象类型
 */
@Data
public class UserQuery extends PageQuery {
    private String keywords;
    private EnableStatus status;
    private Integer deptId;
    private String startTime;
    private String endTime;
}