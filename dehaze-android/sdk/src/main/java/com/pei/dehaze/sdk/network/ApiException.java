package com.pei.dehaze.sdk.network;

import com.pei.dehaze.sdk.model.Result;

import java.io.IOException;
import java.lang.annotation.Annotation;

import okhttp3.ResponseBody;
import retrofit2.Converter;
import retrofit2.Response;
import retrofit2.Retrofit;

/**
 * 自定义异常处理类
 */
public class ApiException extends Exception {
    /** HTTP 状态码或业务错误码的数值形式（用于网络层错误） */
    private final int httpCode;
    /** 业务错误码（String，如 "A0200"、"B0001"），网络层错误时为 null */
    private final String bizCode;
    private final String message;

    public ApiException(int httpCode, String message) {
        this.httpCode = httpCode;
        this.bizCode = null;
        this.message = message;
    }

    public ApiException(int httpCode, String bizCode, String message) {
        this.httpCode = httpCode;
        this.bizCode = bizCode;
        this.message = message;
    }

    /**
     * 解析 HTTP 错误响应体，提取后端返回的业务错误信息
     */
    public static ApiException handleHttpException(Response<?> response, Retrofit retrofit) {
        if (response == null) {
            return new ApiException(-1, "未知错误");
        }

        String message = response.message();
        String bizCode = null;
        try {
            ResponseBody errorBody = response.errorBody();
            if (errorBody != null) {
                Converter<ResponseBody, Result<?>> converter = retrofit.responseBodyConverter(Result.class, new Annotation[0]);
                @SuppressWarnings("unchecked")
                Result<?> result = converter.convert(errorBody);
                if (result != null) {
                    message = result.getMsg();
                    bizCode = result.getCode();
                }
            }
        } catch (IOException e) {
            // 忽略转换异常
        }

        return new ApiException(response.code(), bizCode, message);
    }

    public int getHttpCode() {
        return httpCode;
    }

    public String getBizCode() {
        return bizCode;
    }

    /**
     * 获取业务错误码，若为网络层错误则返回 HTTP 状态码字符串
     */
    public String getCode() {
        return bizCode != null ? bizCode : String.valueOf(httpCode);
    }

    @Override
    public String getMessage() {
        return message;
    }
}
