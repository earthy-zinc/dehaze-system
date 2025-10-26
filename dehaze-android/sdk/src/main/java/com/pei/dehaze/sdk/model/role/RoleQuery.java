package com.pei.dehaze.sdk.model.role;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;

/**
 * 角色查询参数模型类
 */
@Data
public class RoleQuery extends PageQuery {
    private String keywords;
}