package com.pei.dehaze.model.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import com.google.gson.annotations.SerializedName;

/**
 * 注册请求参数模型
 */
@Data
@AllArgsConstructor
public class RegisterRequest {
    @SerializedName("username")
    private String username;
    
    @SerializedName("password")
    private String password;
    
    @SerializedName("email")
    private String email;
    
    @SerializedName("nickname")
    private String nickname;
}