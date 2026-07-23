package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;
import com.pei.dehaze.sdk.utils.TokenManager;

import retrofit2.Call;

/**
 * 认证相关API接口封装
 */
public class AuthAPI {

    private AuthAPI() {
    }

    /**
     * 用户登录
     *
     * @param request  登录请求
     * @param callback 回调
     */
    public static void login(LoginRequest request, ApiCallback<LoginResponse> callback) {
        Call<Result<LoginResponse>> call = DehazeSDK.getInstance().getAuthApiService().login(request);
        call.enqueue(callback);
    }

    /**
     * 获取验证码
     *
     * @param callback 回调
     */
    public static void getCaptcha(ApiCallback<CaptchaResponse> callback) {
        Call<Result<CaptchaResponse>> call = DehazeSDK.getInstance().getAuthApiService().getCaptcha();
        call.enqueue(callback);
    }

    /**
     * 用户注销（清除本地 Token）
     *
     * @param callback 回调
     */
    public static void logout(ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getAuthApiService().logout();
        call.enqueue(new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                // 注销成功后清除本地 Token（accessToken + refreshToken）
                TokenManager.clearAll();
                if (callback != null) {
                    callback.onSuccess(data);
                }
            }

            @Override
            public void onError(String code, String message) {
                if (callback != null) {
                    callback.onError(code, message);
                }
            }

            @Override
            public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                if (callback != null) {
                    callback.onFailure(e);
                }
            }
        });
    }
}
