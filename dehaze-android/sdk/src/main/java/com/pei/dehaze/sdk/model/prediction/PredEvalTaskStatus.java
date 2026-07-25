package com.pei.dehaze.sdk.model.prediction;

/**
 * 预测/评估任务状态枚举
 * <p>
 * 与任务模块的 TaskStatus 区分，专用于预测/评估异步任务。
 */
public enum PredEvalTaskStatus {
    PROCESSING("processing"),
    COMPLETED("completed"),
    FAILED("failed");

    private final String value;

    PredEvalTaskStatus(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    public static PredEvalTaskStatus fromValue(String value) {
        if (value == null) {
            return PROCESSING;
        }
        for (PredEvalTaskStatus status : values()) {
            if (status.value.equalsIgnoreCase(value)) {
                return status;
            }
        }
        return PROCESSING;
    }
}
