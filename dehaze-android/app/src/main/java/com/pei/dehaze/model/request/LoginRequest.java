package com.pei.dehaze.model.request;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * 登录请求参数模型
 */
@Data
@AllArgsConstructor
public class LoginRequest {
    private String username;
    private String password;
    private String captchaCode;
    private String captchaKey;
}