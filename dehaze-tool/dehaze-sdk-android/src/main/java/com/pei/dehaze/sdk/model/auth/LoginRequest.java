package com.pei.dehaze.sdk.model.auth;

import lombok.Data;

/**
 * 登录请求参数模型
 */
@Data
public class LoginRequest {
    private String username;
    private String password;
    private String captchaCode;
    private String captchaKey;
}