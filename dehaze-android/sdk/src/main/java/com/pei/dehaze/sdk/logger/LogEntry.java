package com.pei.dehaze.sdk.logger;

import com.google.gson.JsonObject;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;

/**
 * 单条前端日志条目。
 *
 * 字段规范见前端日志监控改造计划 §3.3（与 JS/Flutter 端对齐，service=client）。
 */
public class LogEntry {

    private final String timestamp;
    private final LogLevel level;
    private final String message;
    private final String app;
    private final String appVersion;
    private String url;
    private String userAgent;
    private String traceId;
    private Integer userId;
    private String errorType;
    private String errorSource;
    private String errorStack;
    private String method;
    private String path;
    private Integer status;
    private Double duration;
    private String code;

    public LogEntry(LogLevel level, String message, String app, String appVersion) {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US);
        sdf.setTimeZone(TimeZone.getTimeZone("UTC"));
        this.timestamp = sdf.format(new Date());
        this.level = level;
        this.message = message;
        this.app = app;
        this.appVersion = appVersion;
    }

    // ==================== setters ====================

    public LogEntry setUrl(String url) {
        this.url = url;
        return this;
    }

    public LogEntry setUserAgent(String userAgent) {
        this.userAgent = userAgent;
        return this;
    }

    public LogEntry setTraceId(String traceId) {
        this.traceId = traceId;
        return this;
    }

    public LogEntry setUserId(Integer userId) {
        this.userId = userId;
        return this;
    }

    public LogEntry setErrorType(String errorType) {
        this.errorType = errorType;
        return this;
    }

    public LogEntry setErrorSource(String errorSource) {
        this.errorSource = errorSource;
        return this;
    }

    public LogEntry setErrorStack(String errorStack) {
        this.errorStack = errorStack;
        return this;
    }

    public LogEntry setMethod(String method) {
        this.method = method;
        return this;
    }

    public LogEntry setPath(String path) {
        this.path = path;
        return this;
    }

    public LogEntry setStatus(Integer status) {
        this.status = status;
        return this;
    }

    public LogEntry setDuration(Double duration) {
        this.duration = duration;
        return this;
    }

    public LogEntry setCode(String code) {
        this.code = code;
        return this;
    }

    // ==================== getters ====================

    public String getTimestamp() { return timestamp; }
    public LogLevel getLevel() { return level; }
    public String getMessage() { return message; }
    public String getApp() { return app; }
    public String getAppVersion() { return appVersion; }
    public String getUrl() { return url; }
    public String getUserAgent() { return userAgent; }
    public String getTraceId() { return traceId; }
    public Integer getUserId() { return userId; }
    public String getErrorType() { return errorType; }
    public String getErrorSource() { return errorSource; }
    public String getErrorStack() { return errorStack; }
    public String getMethod() { return method; }
    public String getPath() { return path; }
    public Integer getStatus() { return status; }
    public Double getDuration() { return duration; }
    public String getCode() { return code; }

    /**
     * 序列化为 NDJSON 对象（仅输出非空字段）。
     */
    public JsonObject toJson() {
        JsonObject obj = new JsonObject();
        obj.addProperty("timestamp", timestamp);
        obj.addProperty("level", level.getLabel());
        obj.addProperty("message", message);
        obj.addProperty("app", app);
        obj.addProperty("app_version", appVersion);
        if (url != null && !url.isEmpty()) obj.addProperty("url", url);
        if (userAgent != null && !userAgent.isEmpty()) obj.addProperty("user_agent", userAgent);
        if (traceId != null && !traceId.isEmpty()) obj.addProperty("trace_id", traceId);
        if (userId != null) obj.addProperty("user_id", userId);
        if (errorType != null) obj.addProperty("error_type", errorType);
        if (errorSource != null) obj.addProperty("error_source", errorSource);
        if (errorStack != null) obj.addProperty("error_stack", errorStack);
        if (method != null) obj.addProperty("method", method);
        if (path != null) obj.addProperty("path", path);
        if (status != null) obj.addProperty("status", status);
        if (duration != null) obj.addProperty("duration", duration);
        if (code != null) obj.addProperty("code", code);
        return obj;
    }
}
