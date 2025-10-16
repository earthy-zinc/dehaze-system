package com.pei.dehaze.sdk.model.user;

import lombok.Data;
import java.util.List;

/**
 * 用户表单类型模型类
 */
@Data
public class UserForm {
    /**
     * 用户头像
     */
    private String avatar;
    
    /**
     * 部门ID
     */
    private Integer deptId;
    
    /**
     * 邮箱
     */
    private String email;
    
    /**
     * 性别
     */
    private Integer gender;
    
    /**
     * 用户ID
     */
    private Integer id;
    
    private String mobile;
    
    /**
     * 昵称
     */
    private String nickname;
    
    /**
     * 角色ID集合
     */
    private List<Integer> roleIds;
    
    /**
     * 用户状态(1:正常;0:禁用)
     */
    private Integer status;
    
    /**
     * 用户名
     */
    private String username;
}