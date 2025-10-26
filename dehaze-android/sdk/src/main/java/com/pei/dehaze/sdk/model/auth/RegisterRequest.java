package com.pei.dehaze.sdk.model.auth;

import lombok.Data;

/**
 * 注册请求参数模型
 */
@Data
public class RegisterRequest {
    private String username;
    private String password;
    private String email;
    private String nickname;
}