package com.pei.dehaze.sdk.api;

import java.util.HashMap;
import java.util.Map;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.auth.AuthInfo;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;
import com.pei.dehaze.sdk.utils.TokenManager;

import retrofit2.Call;

public class AuthAPI {

    private AuthAPI() {
    }

    public static void login(LoginRequest request, ApiCallback<LoginResponse> callback) {
        Call<Result<LoginResponse>> call = DehazeSDK.getInstance().getAuthApiService().login(request);
        call.enqueue(callback);
    }

    public static void register(LoginRequest request, ApiCallback<LoginResponse> callback) {
        Map<String, Object> body = new HashMap<>();
        body.put("username", request.getUsername());
        body.put("password", request.getPassword());
        body.put("captchaKey", request.getCaptchaKey());
        body.put("captchaCode", request.getCaptchaCode());
        body.put("nickname", request.getNickname());
        Call<Result<LoginResponse>> call = DehazeSDK.getInstance().getAuthApiService().register(body);
        call.enqueue(callback);
    }

    public static void getCaptcha(ApiCallback<CaptchaResponse> callback) {
        Call<Result<CaptchaResponse>> call = DehazeSDK.getInstance().getAuthApiService().getCaptcha();
        call.enqueue(callback);
    }

    public static void logout(ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getAuthApiService().logout();
        call.enqueue(new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
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

    public static void getAuthInfo(ApiCallback<AuthInfo> callback) {
        Call<Result<AuthInfo>> call = DehazeSDK.getInstance().getAuthApiService().getAuthInfo();
        call.enqueue(callback);
    }
}
