package com.pei.dehaze.sdk.model.auth;

import lombok.Data;

@Data
public class LoginRequest {
    private String username;
    private String password;
    private String nickname;
    private String captchaCode;
    private String captchaKey;
    private Boolean rememberMe;
}
