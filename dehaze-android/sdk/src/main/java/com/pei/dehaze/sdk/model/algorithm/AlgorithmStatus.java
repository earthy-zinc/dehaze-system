package com.pei.dehaze.sdk.model.algorithm;

/**
 * 算法状态机枚举
 * 6状态：草稿0 → 测试中1 → 待审核2 → 已发布3 → 已停用4 → 已归档5
 */
public enum AlgorithmStatus {
    DRAFT(0, "草稿"),
    TESTING(1, "测试中"),
    PENDING_AUDIT(2, "待审核"),
    PUBLISHED(3, "已发布"),
    DISABLED(4, "已停用"),
    ARCHIVED(5, "已归档");

    private final int value;
    private final String label;

    AlgorithmStatus(int value, String label) {
        this.value = value;
        this.label = label;
    }

    public int getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    public static AlgorithmStatus fromValue(int value) {
        for (AlgorithmStatus status : values()) {
            if (status.value == value) {
                return status;
            }
        }
        return DRAFT;
    }
}
