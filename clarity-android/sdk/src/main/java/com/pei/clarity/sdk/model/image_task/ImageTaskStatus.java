package com.pei.clarity.sdk.model.image_task;

/**
 * 图像任务状态枚举
 * 对齐后端 LogStatusEnum：1=处理中，2=已完成，3=失败，4=已取消
 */
public enum ImageTaskStatus {
    PROCESSING(1, "处理中"),
    COMPLETED(2, "已完成"),
    FAILED(3, "失败"),
    CANCELED(4, "已取消");

    private final int value;
    private final String label;

    ImageTaskStatus(int value, String label) {
        this.value = value;
        this.label = label;
    }

    public int getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    public static ImageTaskStatus fromValue(Integer value) {
        if (value == null) return null;
        for (ImageTaskStatus status : values()) {
            if (status.value == value) {
                return status;
            }
        }
        return null;
    }
}
