package com.pei.dehaze.sdk.model.auth;

import lombok.Data;

/**
 * 登录请求参数模型
 * 对齐后端 LoginRequest：username、password、captchaCode、captchaKey
 */
@Data
public class LoginRequest {
    private String username;
    private String password;
    private String captchaCode;
    private String captchaKey;
}
