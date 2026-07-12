package com.pei.dehaze.sdk.model.auth;

import java.util.List;

import lombok.Data;

/**
 * 当前用户认证信息
 * 对齐后端 UserInfoVO：userId、username、nickname、avatar、roles、perms
 * 用于 GET /api/v1/auth/me 响应
 */
@Data
public class AuthInfo {
    /** 用户ID */
    private long userId;
    /** 用户名 */
    private String username;
    /** 昵称 */
    private String nickname;
    /** 头像地址 */
    private String avatar;
    /** 角色编码集合 */
    private List<String> roles;
    /** 权限标识集合 */
    private List<String> perms;
}
