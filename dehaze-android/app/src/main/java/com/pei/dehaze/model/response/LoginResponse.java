package com.pei.dehaze.model.response;

import lombok.Data;
import com.google.gson.annotations.SerializedName;

/**
 * 登录响应数据模型
 */
@Data
public class LoginResponse {
    @SerializedName("token")
    private String token;

    @SerializedName("tokenType")
    private String tokenType;

    @SerializedName("expiresIn")
    private long expiresIn;

    @SerializedName("userInfo")
    private UserInfo userInfo;

    /**
     * 用户信息数据结构
     */
    @Data
    public static class UserInfo {
        @SerializedName("id")
        private String id;

        @SerializedName("username")
        private String username;

        @SerializedName("nickname")
        private String nickname;

        @SerializedName("email")
        private String email;

        @SerializedName("avatar")
        private String avatar;

        @SerializedName("createdAt")
        private String createdAt;
    }
}