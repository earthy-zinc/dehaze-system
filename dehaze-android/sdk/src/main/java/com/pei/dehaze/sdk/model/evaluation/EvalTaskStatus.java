package com.pei.dehaze.sdk.model.evaluation;

/**
 * 评估任务状态枚举
 * 对齐后端 LogStatusEnum：1=处理中，2=已完成，3=失败
 */
public enum EvalTaskStatus {
    PROCESSING(1, "处理中"),
    COMPLETED(2, "已完成"),
    FAILED(3, "失败");

    private final int value;
    private final String label;

    EvalTaskStatus(int value, String label) {
        this.value = value;
        this.label = label;
    }

    public int getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    public static EvalTaskStatus fromValue(Integer value) {
        if (value == null) return null;
        for (EvalTaskStatus status : values()) {
            if (status.value == value) {
                return status;
            }
        }
        return null;
    }
}
