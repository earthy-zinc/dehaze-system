package com.pei.dehaze.model.response;

import lombok.Data;
import com.google.gson.annotations.SerializedName;

/**
 * 获取用户信息响应模型
 */
@Data
public class UserInfoResponse {
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
    
    @SerializedName("updatedAt")
    private String updatedAt;
}