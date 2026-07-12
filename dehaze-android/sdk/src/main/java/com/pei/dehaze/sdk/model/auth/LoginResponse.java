package com.pei.dehaze.sdk.model.auth;

import lombok.Data;

/**
 * 登录响应数据模型
 * 对齐后端 LoginResult：accessToken、tokenType、refreshToken、expires、user
 */
@Data
public class LoginResponse {
    /** 访问令牌 */
    private String accessToken;
    /** Token 类型 */
    private String tokenType;
    /** 刷新令牌 */
    private String refreshToken;
    /** 过期时间（毫秒） */
    private long expires;
    /** 用户信息 */
    private LoginUser user;

    /**
     * 登录返回的用户信息
     */
    @Data
    public static class LoginUser {
        private long id;
        private String username;
        private String nickname;
    }
}
