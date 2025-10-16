package com.pei.dehaze.sdk.utils;

import com.pei.dehaze.sdk.network.ApiException;

/**
 * 错误处理工具类
 * 提供详细的错误信息解析和处理功能
 */
public class ErrorUtils {
    
    /**
     * 解析网络错误
     *
     * @param e ApiException实例
     * @return 格式化后的错误信息
     */
    public static String parseError(ApiException e) {
        if (e == null) {
            return "未知错误";
        }
        
        int code = e.getCode();
        String message = e.getMessage();
        
        // 根据错误码返回更友好的错误信息
        switch (code) {
            case 400:
                return "请求参数错误";
            case 401:
                return "未授权访问，请重新登录";
            case 403:
                return "访问被禁止";
            case 404:
                return "请求的资源不存在";
            case 500:
                return "服务器内部错误";
            case -1:
                return "网络连接失败，请检查网络设置";
            default:
                if (message != null && !message.isEmpty()) {
                    return message;
                } else {
                    return "未知错误 (错误码: " + code + ")";
                }
        }
    }
    
    /**
     * 根据HTTP状态码生成错误信息
     *
     * @param code HTTP状态码
     * @return 错误信息
     */
    public static String getErrorMessageByCode(int code) {
        switch (code) {
            case 400:
                return "请求参数错误";
            case 401:
                return "未授权访问";
            case 403:
                return "访问被禁止";
            case 404:
                return "请求的资源不存在";
            case 500:
                return "服务器内部错误";
            default:
                return "HTTP错误 " + code;
        }
    }
}