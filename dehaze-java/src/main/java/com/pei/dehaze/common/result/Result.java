package com.pei.dehaze.common.result;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import org.slf4j.MDC;

import java.io.Serializable;

/**
 * 统一响应结构体
 *
 * @author earthyzinc
 * @since 2022/1/30
 **/
@Data
public class Result<T> implements Serializable {
    @Schema(description = "错误码")
    private String code;

    @Schema(description = "返回数据")
    private T data;

    @Schema(description = "消息")
    private String msg;

    @Schema(description = "请求追踪ID，用于问题排查")
    private String traceId;

    public static <T> Result<T> success() {
        return success(null);
    }

    public static <T> Result<T> success(T data) {
        Result<T> result = new Result<>();
        result.setCode(ResultCode.SUCCESS.getCode());
        result.setMsg(ResultCode.SUCCESS.getMsg());
        result.setData(data);
        result.setTraceId(MDC.get("trace_id"));
        return result;
    }

    public static <T> Result<T> failed() {
        return result(ResultCode.SYSTEM_EXECUTION_ERROR.getCode(), ResultCode.SYSTEM_EXECUTION_ERROR.getMsg(), null);
    }

    public static <T> Result<T> failed(String msg) {
        return result(ResultCode.SYSTEM_EXECUTION_ERROR.getCode(), msg, null);
    }

    public static <T> Result<T> judge(boolean status) {
        if (status) {
            return success();
        } else {
            return failed();
        }
    }

    public static <T> Result<T> failed(IResultCode resultCode) {
        return result(resultCode.getCode(), resultCode.getMsg(), null);
    }

    public static <T> Result<T> failed(IResultCode resultCode, String msg) {
        return result(resultCode.getCode(), msg, null);
    }

    private static <T> Result<T> result(String code, String msg, T data) {
        Result<T> result = new Result<>();
        result.setCode(code);
        result.setData(data);
        result.setMsg(msg);
        result.setTraceId(MDC.get("trace_id"));
        return result;
    }
}
