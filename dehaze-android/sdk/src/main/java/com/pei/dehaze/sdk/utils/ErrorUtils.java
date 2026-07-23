package com.pei.dehaze.sdk.utils;

import com.pei.dehaze.sdk.network.ApiException;

/**
 * 错误处理工具类
 * 提供详细的错误信息解析和处理功能
 */
public class ErrorUtils {

    /**
     * 解析网络错误为友好的提示信息
     *
     * @param e ApiException实例
     * @return 格式化后的错误信息
     */
    public static String parseError(ApiException e) {
        if (e == null) {
            return "未知错误";
        }

        // 优先使用业务错误码判断
        String bizCode = e.getBizCode();
        if (bizCode != null) {
            return getMessageByBizCode(bizCode, e.getMessage());
        }

        // 网络层 HTTP 错误
        return getMessageByHttpCode(e.getHttpCode(), e.getMessage());
    }

    /**
     * 根据业务错误码返回友好提示
     */
    private static String getMessageByBizCode(String code, String defaultMessage) {
        // 仅处理常见的需要特殊提示的错误码，其他直接返回后端消息
        switch (code) {
            case "A0200":
                return "用户登录异常";
            case "A0210":
                return "用户名或密码错误";
            case "A0213":
                return "验证码已过期，请刷新";
            case "A0214":
                return "验证码错误";
            case "A0230":
            case "A0231":
                return "登录已过期，请重新登录";
            case "A0301":
                return "访问未授权";
            case "A0400":
            case "A0410":
                return "请求参数错误";
            case "A0401":
                return "请求的资源不存在";
            case "A0501":
                return "数据已存在";
            case "A0502":
                return "数据状态不允许此操作";
            case "A0504":
                return "存在关联数据，无法删除";
            case "B0001":
                return "服务器内部错误";
            case "B0210":
            case "B0211":
                return "系统繁忙，请稍后重试";
            default:
                return defaultMessage != null && !defaultMessage.isEmpty() ? defaultMessage : "操作失败 (" + code + ")";
        }
    }

    /**
     * 根据HTTP状态码生成错误信息
     */
    private static String getMessageByHttpCode(int httpCode, String defaultMessage) {
        switch (httpCode) {
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
            case 0:
                return "网络连接失败，请检查网络设置";
            default:
                return defaultMessage != null && !defaultMessage.isEmpty() ? defaultMessage : "网络错误 (" + httpCode + ")";
        }
    }
}
