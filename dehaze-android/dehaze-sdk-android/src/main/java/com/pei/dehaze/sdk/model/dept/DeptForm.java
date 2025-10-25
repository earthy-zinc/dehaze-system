package com.pei.dehaze.sdk.model.dept;

import lombok.Data;

/**
 * 部门表单模型类
 */
@Data
public class DeptForm {
    /**
     * 部门ID(新增不填)
     */
    private Integer id;
    
    /**
     * 部门名称
     */
    private String name;
    
    /**
     * 父部门ID
     */
    private int parentId;
    
    /**
     * 排序
     */
    private Integer sort;
    
    /**
     * 状态(1:启用；0：禁用)
     */
    private Integer status;
}