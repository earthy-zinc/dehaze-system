package com.pei.dehaze.sdk;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.network.ApiException;
import com.pei.dehaze.sdk.utils.TokenManager;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * API回调包装类，用于简化API调用的处理
 */
public abstract class ApiCallback<T> implements Callback<Result<T>> {

    /**
     * 请求成功且业务逻辑成功时调用
     *
     * @param data 业务数据
     */
    public abstract void onSuccess(T data);

    /**
     * 请求成功但业务逻辑失败时调用
     *
     * @param code    业务错误码（如 "A0200"、"B0001"）
     * @param message 错误消息
     */
    public void onError(String code, String message) {
        // 默认空实现
    }

    /**
     * 网络请求失败时调用（网络异常或HTTP错误）
     *
     * @param e 异常信息
     */
    public void onFailure(ApiException e) {
        // 默认空实现
    }

    @Override
    public void onResponse(Call<Result<T>> call, Response<Result<T>> response) {
        if (response.isSuccessful()) {
            Result<T> result = response.body();
            if (result == null) {
                onFailure(new ApiException(response.code(), "响应体为空"));
                return;
            }
            if (result.isSuccess()) {
                onSuccess(result.getData());
            } else {
                // token 无效业务码，清除本地全部 token（accessToken + refreshToken）
                if (TokenManager.isTokenInvalidCode(result.getCode())) {
                    TokenManager.clearAll();
                }
                onError(result.getCode(), result.getMsg());
            }
        } else {
            // HTTP 错误，解析后端返回的业务错误信息
            ApiException exception = ApiException.handleHttpException(response, DehazeSDK.getInstance().getRetrofit());
            // 401（token 过期/无效，且 OkHttp Authenticator 刷新失败或不可用）清除全部 token
            if (response.code() == 401) {
                TokenManager.clearAll();
            }
            onFailure(exception);
        }
    }

    @Override
    public void onFailure(Call<Result<T>> call, Throwable t) {
        onFailure(new ApiException(0, t.getMessage()));
    }
}
