package com.pei.dehaze.network;

import com.pei.dehaze.model.response.CaptchaResponse;
import com.pei.dehaze.model.request.LoginRequest;
import com.pei.dehaze.model.response.LoginResponse;
import com.pei.dehaze.model.request.RegisterRequest;
import com.pei.dehaze.model.response.RegisterResponse;
import com.pei.dehaze.common.Result;
import com.pei.dehaze.model.response.UserInfoResponse;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Query;

/**
 * API服务接口，定义所有的网络请求方法
 */
public interface ApiService {

    /**
     * 用户登录接口
     */
    @POST("auth/login")
    Call<Result<LoginResponse>> login(
            @Query("username") String username,
            @Query("password") String password,
            @Query("captchaCode") String captchaCode,
            @Query("captchaKey") String captchaKey);

    /**
     * 用户注册接口
     */
    @POST("auth/register")
    Call<Result<RegisterResponse>> register(@Body RegisterRequest registerRequest);

    /**
     * 获取用户信息接口
     */
    @GET("user/info")
    Call<Result<UserInfoResponse>> getUserInfo();
    
    /**
     * 获取验证码接口
     */
    @GET("auth/captcha")
    Call<Result<CaptchaResponse>> getCaptcha();
}