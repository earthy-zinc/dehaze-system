package com.pei.clarity.sdk.service;

import com.pei.clarity.sdk.model.Result;
import com.pei.clarity.sdk.model.auth.AuthInfo;
import com.pei.clarity.sdk.model.auth.CaptchaResponse;
import com.pei.clarity.sdk.model.auth.LoginRequest;
import com.pei.clarity.sdk.model.auth.LoginResponse;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;

public interface AuthApiService {
    @POST("/api/v1/auth/login")
    Call<Result<LoginResponse>> login(@Body LoginRequest request);

    @POST("/api/v1/auth/register")
    Call<Result<LoginResponse>> register(@Body LoginRequest request);

    @GET("/api/v1/auth/captcha")
    Call<Result<CaptchaResponse>> getCaptcha();

    @POST("/api/v1/auth/logout")
    Call<Result<Void>> logout();

    @GET("/api/v1/auth/me")
    Call<Result<AuthInfo>> getAuthInfo();
}
