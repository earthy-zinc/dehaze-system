package com.pei.dehaze.sdk.model.role;

import com.pei.dehaze.sdk.model.EnableStatus;
import lombok.Data;
import java.util.Date;

/**
 * 角色分页对象模型类
 */
@Data
public class RolePageVO {
    /**
     * 角色编码
     */
    private String code;
    
    /**
     * 角色ID
     */
    private Integer id;
    
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
    
    /**
     * 创建时间
     */
    private Date createTime;
    
    /**
     * 修改时间
     */
    private Date updateTime;
}