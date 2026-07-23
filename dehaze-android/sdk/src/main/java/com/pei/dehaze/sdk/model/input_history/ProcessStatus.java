package com.pei.dehaze.sdk.model.input_history;

/**
 * 图像处理状态枚举
 * 对齐后端 InputHistory.status：1=成功，2=失败，3=处理中
 */
public enum ProcessStatus {
    SUCCESS(1, "成功"),
    FAILED(2, "失败"),
    PROCESSING(3, "处理中");

    private final int value;
    private final String label;

    ProcessStatus(int value, String label) {
        this.value = value;
        this.label = label;
    }

    public int getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    public static ProcessStatus fromValue(Integer value) {
        if (value == null) return null;
        for (ProcessStatus status : values()) {
            if (status.value == value) {
                return status;
            }
        }
        return null;
    }
}
