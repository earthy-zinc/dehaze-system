package com.pei.dehaze.sdk.model.auth;

import lombok.Data;

/**
 * 登录响应数据模型
 */
@Data
public class LoginResponse {
    private String token;
    private String tokenType;
    private long expiresIn;
    private UserInfo userInfo;

    /**
     * 用户信息数据结构
     */
    @Data
    public static class UserInfo {
        private String id;
        private String username;
        private String nickname;
        private String email;
        private String avatar;
        private String createdAt;
    }
}