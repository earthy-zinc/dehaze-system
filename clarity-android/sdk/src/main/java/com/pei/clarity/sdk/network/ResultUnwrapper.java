package com.pei.clarity.sdk.network;

import com.pei.clarity.sdk.model.Result;

import java.io.IOException;

import retrofit2.Response;

/**
 * 轮询场景下的同步响应解包：供需要自行轮询的 API 门面复用。
 */
public final class ResultUnwrapper {

    private ResultUnwrapper() {
    }

    /**
     * 同步执行 HTTP 请求并解包 Result 容器，失败时抛出 ApiException
     */
    public static <T> T unwrap(Response<Result<T>> response) throws ApiException, IOException {
        if (response == null || !response.isSuccessful()) {
            int code = response == null ? -1 : response.code();
            throw new ApiException(code, "HTTP " + code);
        }
        Result<T> body = response.body();
        if (body == null) {
            throw new ApiException(0, "响应体为空");
        }
        if (!body.isSuccess()) {
            throw new ApiException(0, body.getCode(), body.getMsg());
        }
        return body.getData();
    }
}
