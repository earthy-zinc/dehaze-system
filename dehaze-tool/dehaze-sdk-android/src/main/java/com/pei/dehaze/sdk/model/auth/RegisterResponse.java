package com.pei.dehaze.sdk.model.auth;

import lombok.Data;

/**
 * 注册响应数据模型
 */
@Data
public class RegisterResponse {
    private String id;
    private String username;
    private String email;
    private String nickname;
    private String createdAt;
}