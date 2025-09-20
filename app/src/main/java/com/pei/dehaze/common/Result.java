package com.pei.dehaze.common;

import lombok.Data;
import com.google.gson.annotations.SerializedName;

/**
 * 通用响应类，用于封装所有接口的响应数据
 * @param <T> 数据类型
 */
@Data
public class Result<T> {
    @SerializedName("code")
    private String code; // 响应码，"00000"表示成功
    
    @SerializedName("msg")
    private String msg; // 响应消息
    
    @SerializedName("data")
    private T data; // 响应数据
}
