package com.pei.dehaze.model.response;

import lombok.Data;
import com.google.gson.annotations.SerializedName;

/**
 * 注册响应数据模型
 */
@Data
public class RegisterResponse {
    @SerializedName("userId")
    private String userId;
    
    @SerializedName("username")
    private String username;
    
    @SerializedName("createdAt")
    private String createdAt;
}