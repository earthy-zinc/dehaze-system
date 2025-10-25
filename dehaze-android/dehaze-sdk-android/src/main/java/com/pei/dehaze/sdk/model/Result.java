package com.pei.dehaze.sdk.model;

import lombok.Data;

/**
 * 通用响应结果包装类
 */
@Data
public class Result<T> {
    private int code;
    private T data;
    private String message;

    public boolean isSuccess() {
        return code == 0;
    }
}