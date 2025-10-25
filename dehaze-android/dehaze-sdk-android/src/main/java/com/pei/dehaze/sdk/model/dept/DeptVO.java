package com.pei.dehaze.sdk.model.dept;

import lombok.Data;
import java.util.Date;
import java.util.List;

/**
 * 部门模型类
 */
@Data
public class DeptVO {
    /**
     * 子部门
     */
    private List<DeptVO> children;
    
    /**
     * 创建时间
     */
    private Date createTime;
    
    /**
     * 部门ID
     */
    private Integer id;
    
    /**
     * 部门名称
     */
    private String name;
    
    /**
     * 父部门ID
     */
    private Integer parentId;
    
    /**
     * 排序
     */
    private Integer sort;
    
    /**
     * 状态(1:启用；0:禁用)
     */
    private Integer status;
    
    /**
     * 修改时间
     */
    private Date updateTime;
}