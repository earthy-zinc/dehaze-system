package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;
import com.pei.dehaze.sdk.model.auth.RegisterRequest;
import com.pei.dehaze.sdk.model.auth.RegisterResponse;

import retrofit2.Call;

/**
 * 认证相关API接口封装
 */
public class AuthAPI {

    /**
     * 用户登录
     *
     * @param username    用户名
     * @param password    密码
     * @param captchaCode 验证码
     * @param captchaKey  验证码key
     * @param callback    回调函数
     */
    public static void login(String username, String password, String captchaCode, String captchaKey, ApiCallback<LoginResponse> callback) {
        Call<Result<LoginResponse>> call = DehazeSDK.getInstance().getAuthApiService().login(username, password, captchaCode, captchaKey);
        call.enqueue(callback);
    }

    /**
     * 用户登录（使用LoginRequest对象）
     *
     * @param request  登录请求对象
     * @param callback 回调函数
     */
    public static void login(LoginRequest request, ApiCallback<LoginResponse> callback) {
        login(request.getUsername(), request.getPassword(), request.getCaptchaCode(), request.getCaptchaKey(), callback);
    }

    /**
     * 用户注册
     *
     * @param request  注册请求对象
     * @param callback 回调函数
     */
    public static void register(RegisterRequest request, ApiCallback<RegisterResponse> callback) {
        Call<Result<RegisterResponse>> call = DehazeSDK.getInstance().getAuthApiService().register(request);
        call.enqueue(callback);
    }

    /**
     * 获取验证码
     *
     * @param callback 回调函数
     */
    public static void getCaptcha(ApiCallback<CaptchaResponse> callback) {
        Call<Result<CaptchaResponse>> call = DehazeSDK.getInstance().getAuthApiService().getCaptcha();
        call.enqueue(callback);
    }
}