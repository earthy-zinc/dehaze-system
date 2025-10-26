package com.pei.dehaze.sdk;

import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.network.ApiException;
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
     * @param code    错误码
     * @param message 错误消息
     */
    public void onError(int code, String message) {
        // 默认空实现
    }

    /**
     * 网络请求失败时调用
     *
     * @param e 异常信息
     */
    public void onFailure(ApiException e) {
        // 默认空实现
    }

    @Override
    public void onResponse(Call<Result<T>> call, Response<Result<T>> response) {
        if (response.isSuccessful() && response.body() != null) {
            Result<T> result = response.body();
            if (result.isSuccess()) {
                onSuccess(result.getData());
            } else {
                onError(result.getCode(), result.getMessage());
            }
        } else {
            onFailure(new ApiException(response.code(), "Response not successful or body is null"));
        }
    }

    @Override
    public void onFailure(Call<Result<T>> call, Throwable t) {
        onFailure(new ApiException(0, t.getMessage()));
    }
}