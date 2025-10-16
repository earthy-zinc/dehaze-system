package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginResponse;
import com.pei.dehaze.sdk.model.auth.RegisterRequest;
import com.pei.dehaze.sdk.model.auth.RegisterResponse;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Query;

/**
 * 认证相关API服务接口
 */
public interface AuthApiService {
    /**
     * 用户登录接口
     */
    @POST("/api/v1/auth/login")
    Call<Result<LoginResponse>> login(
            @Query("username") String username,
            @Query("password") String password,
            @Query("captchaCode") String captchaCode,
            @Query("captchaKey") String captchaKey);

    /**
     * 用户注册接口
     */
    @POST("/api/v1/auth/register")
    Call<Result<RegisterResponse>> register(@Body RegisterRequest registerRequest);

    /**
     * 获取验证码接口
     */
    @GET("/api/v1/auth/captcha")
    Call<Result<CaptchaResponse>> getCaptcha();
}
