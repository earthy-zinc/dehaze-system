package com.pei.dehaze.sdk.logger;

/**
 * 日志级别（与前端日志监控改造计划 §3.3 对齐）。
 */
public enum LogLevel {
    ERROR("ERROR"),
    WARN("WARN"),
    INFO("INFO");

    private final String label;

    LogLevel(String label) {
        this.label = label;
    }

    public String getLabel() {
        return label;
    }
}
