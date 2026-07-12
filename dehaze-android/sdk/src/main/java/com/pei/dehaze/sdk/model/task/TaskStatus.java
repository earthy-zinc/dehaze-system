package com.pei.dehaze.sdk.model.task;

/**
 * 任务状态枚举
 * 对齐后端任务状态机：PENDING → PROCESSING → COMPLETED/FAILED/CANCELLED
 */
public enum TaskStatus {
    PENDING("pending", "等待中"),
    PROCESSING("processing", "处理中"),
    COMPLETED("completed", "已完成"),
    FAILED("failed", "失败"),
    CANCELLED("cancelled", "已取消");

    private final String value;
    private final String label;

    TaskStatus(String value, String label) {
        this.value = value;
        this.label = label;
    }

    public String getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    public static TaskStatus fromValue(String value) {
        if (value == null) {
            return PENDING;
        }
        for (TaskStatus status : values()) {
            if (status.value.equalsIgnoreCase(value)) {
                return status;
            }
        }
        return PENDING;
    }
}
