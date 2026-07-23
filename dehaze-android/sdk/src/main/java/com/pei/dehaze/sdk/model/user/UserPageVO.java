package com.pei.dehaze.sdk.model.user;

import com.pei.dehaze.sdk.model.EnableStatus;
import lombok.Data;
import java.util.Date;

/**
 * 用户分页对象模型类
 */
@Data
public class UserPageVO {
    /**
     * 用户头像地址
     */
    private String avatar;
    
    /**
     * 创建时间
     */
    private Date createTime;
    
    /**
     * 部门名称
     */
    private String deptName;
    
    /**
     * 用户邮箱
     */
    private String email;
    
    /**
     * 性别
     */
    private String genderLabel;
    
    /**
     * 用户ID
     */
    private Integer id;
    
    /**
     * 手机号
     */
    private String mobile;
    
    /**
     * 用户昵称
     */
    private String nickname;
    
    /**
     * 角色名称，多个使用英文逗号(,)分割
     */
    private String roleNames;
    
    /**
     * 用户状态
     */
    private EnableStatus status;
    
    /**
     * 用户名
     */
    private String username;
}