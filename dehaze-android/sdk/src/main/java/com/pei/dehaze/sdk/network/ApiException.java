package com.pei.dehaze.sdk.network;

import com.pei.dehaze.sdk.model.Result;

import java.io.IOException;

import okhttp3.ResponseBody;
import retrofit2.Converter;
import retrofit2.Response;
import retrofit2.Retrofit;

/**
 * 自定义异常处理类
 */
public class ApiException extends Exception {
    private final int code;
    private final String message;

    public ApiException(int code, String message) {
        this.code = code;
        this.message = message;
    }

    public static ApiException handleHttpException(Response<?> response, Retrofit retrofit) {
        if (response == null) {
            return new ApiException(-1, "未知错误");
        }

        String message = response.message();
        try {
            ResponseBody errorBody = response.errorBody();
            if (errorBody != null) {
                Converter<ResponseBody, Result> converter = retrofit.responseBodyConverter(Result.class, new java.lang.annotation.Annotation[0]);
                Result result = converter.convert(errorBody);
                if (result != null) {
                    message = result.getMessage();
                }
            }
        } catch (IOException e) {
            // 忽略转换异常
        }

        return new ApiException(response.code(), message);
    }

    public int getCode() {
        return code;
    }

    @Override
    public String getMessage() {
        return message;
    }
}