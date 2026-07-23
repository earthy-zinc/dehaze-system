package com.pei.dehaze.sdk.model.role;

import com.pei.dehaze.sdk.model.EnableStatus;
import lombok.Data;

/**
 * 角色表单对象模型类
 */
@Data
public class RoleForm {
    /**
     * 角色ID
     */
    private Integer id;
    
    /**
     * 角色编码
     */
    private String code;
    
    /**
     * 数据权限
     */
    private Integer dataScope;
    
    /**
     * 角色名称
     */
    private String name;
    
    /**
     * 排序
     */
    private Integer sort;
    
    /**
     * 角色状态
     */
    private EnableStatus status;
}