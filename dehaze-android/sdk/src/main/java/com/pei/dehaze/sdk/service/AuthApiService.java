package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.auth.AuthInfo;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.Header;
import retrofit2.http.POST;

/**
 * 认证相关API服务接口
 * 对齐后端路由：/api/v1/auth（login、captcha 为公开接口；logout、me、refresh 需认证）
 */
public interface AuthApiService {
    /**
     * 用户登录
     * POST /api/v1/auth/login
     */
    @POST("/api/v1/auth/login")
    Call<Result<LoginResponse>> login(@Body LoginRequest request);

    /**
     * 获取验证码
     * GET /api/v1/auth/captcha
     */
    @GET("/api/v1/auth/captcha")
    Call<Result<CaptchaResponse>> getCaptcha();

    /**
     * 用户注销
     * POST /api/v1/auth/logout
     */
    @POST("/api/v1/auth/logout")
    Call<Result<Void>> logout();

    /**
     * 获取当前登录用户信息（昵称、头像、权限、角色）
     * GET /api/v1/auth/me
     */
    @GET("/api/v1/auth/me")
    Call<Result<AuthInfo>> getAuthInfo();

    /**
     * 刷新 Token
     * POST /api/v1/auth/refresh
     *
     * @param refreshToken 刷新令牌
     */
    @POST("/api/v1/auth/refresh")
    Call<Result<LoginResponse>> refreshToken(@Header("refreshToken") String refreshToken);
}
