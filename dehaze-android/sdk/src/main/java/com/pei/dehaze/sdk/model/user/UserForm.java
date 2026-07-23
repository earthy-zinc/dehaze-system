package com.pei.dehaze.sdk.model.user;

import com.pei.dehaze.sdk.model.EnableStatus;
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
    private Gender gender;

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
     * 用户状态
     */
    private EnableStatus status;
    
    /**
     * 用户名
     */
    private String username;
}