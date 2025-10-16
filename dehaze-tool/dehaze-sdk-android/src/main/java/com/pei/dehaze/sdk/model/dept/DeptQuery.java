package com.pei.dehaze.sdk.model.dept;

import lombok.Data;

/**
 * 部门查询参数模型类
 */
@Data
public class DeptQuery {
    private String keywords;
    private Integer status;
}