package com.pei.dehaze.sdk.model;

import java.util.List;

import lombok.Data;

/**
 * 通用响应结果包装类
 * 对齐后端 Response 结构：code(String, "00000"表示成功)、msg、data、traceId、timestamp、errors
 */
@Data
public class Result<T> {
    /** 响应码，"00000" 表示成功 */
    private String code;
    /** 响应数据 */
    private T data;
    /** 响应消息 */
    private String msg;
    /** 链路追踪ID */
    private String traceId;
    /** 时间戳 */
    private Long timestamp;
    /** 参数校验错误列表 */
    private List<ErrorItem> errors;

    /** 成功状态码 */
    public static final String CODE_SUCCESS = "00000";

    public boolean isSuccess() {
        return CODE_SUCCESS.equals(code);
    }

    /**
     * 参数校验错误项
     */
    @Data
    public static class ErrorItem {
        private String field;
        private String message;
        private String code;
    }
}
