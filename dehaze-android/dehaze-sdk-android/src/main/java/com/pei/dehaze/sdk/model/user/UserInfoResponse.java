package com.pei.dehaze.sdk.model.user;

import lombok.Data;

/**
 * 获取用户信息响应模型
 */
@Data
public class UserInfoResponse {
    private String id;
    private String username;
    private String nickname;
    private String email;
    private String avatar;
    private String createdAt;
    private String updatedAt;
}