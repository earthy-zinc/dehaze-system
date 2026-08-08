package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 前端单条日志条目（由前端 SDK 采集上报）。
 * <p>
 * 字段规范见 dehaze-doc/docs/02-系统架构/07-日志架构设计.md §3.5 与
 * dehaze-doc/docs/05-改造计划/前端日志监控改造计划.md §3.3。
 */
@Data
@Schema(description = "前端日志条目")
public class ClientLogEntryForm {

    @Schema(description = "ISO8601 UTC 时间戳")
    private String timestamp;

    @Schema(description = "日志级别: ERROR/WARN/INFO")
    private String level;

    @Schema(description = "人读描述")
    private String message;

    @Schema(description = "前端项目标识: react/vue/taro/uniapp/rn/flutter/android")
    private String app;

    @Schema(description = "应用版本号")
    private String appVersion;

    @Schema(description = "当前页面 URL / 路由路径")
    private String url;

    @Schema(description = "浏览器/设备 User-Agent")
    private String userAgent;

    @Schema(description = "请求追踪ID（与后端日志串联，匿名上报必带）")
    private String traceId;

    @Schema(description = "错误类型: js/dart/native/promise/api")
    private String errorType;

    @Schema(description = "错误来源: window.onerror/unhandledrejection/FlutterError/api_interceptor")
    private String errorSource;

    @Schema(description = "完整堆栈字符串")
    private String errorStack;

    @Schema(description = "HTTP 方法（API 失败日志）")
    private String method;

    @Schema(description = "请求路径（API 失败日志，不含 query）")
    private String path;

    @Schema(description = "HTTP 状态码（API 失败日志）")
    private Integer status;

    @Schema(description = "请求耗时毫秒（API 失败日志）")
    private Double duration;

    @Schema(description = "业务错误码（API 失败日志）")
    private String code;

    @Schema(description = "日志类型: 普通/performance")
    private String type;

    @Schema(description = "性能指标名: lcp/inp/cls/fcp/fp/ttfb/dom_ready/load/long_task/resource_error/route_switch")
    private String metricName;

    @Schema(description = "性能指标值")
    private Double metricValue;

    @Schema(description = "导航类型: navigate/reload/back_forward")
    private String navigationType;

    @Schema(description = "资源 URL")
    private String resourceUrl;
}
