package com.pei.dehaze.sdk.model.task;

/**
 * 任务类型枚举
 * 对齐后端 TaskConstants：dataset_export、item_download、batch_download、custom_export
 */
public enum TaskType {
    DATASET_EXPORT("dataset_export", "数据集导出"),
    ITEM_DOWNLOAD("item_download", "数据项下载"),
    BATCH_DOWNLOAD("batch_download", "批量下载"),
    CUSTOM_EXPORT("custom_export", "自定义导出");

    private final String value;
    private final String label;

    TaskType(String value, String label) {
        this.value = value;
        this.label = label;
    }

    public String getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }
}
