package com.pei.dehaze.sdk.model;

/**
 * 启用/禁用状态枚举
 * 对齐后端 User/Role.status：0=禁用，1=启用
 */
public enum EnableStatus {
    DISABLED(0, "禁用"),
    ENABLED(1, "启用");

    private final int value;
    private final String label;

    EnableStatus(int value, String label) {
        this.value = value;
        this.label = label;
    }

    public int getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    public static EnableStatus fromValue(Integer value) {
        if (value == null) return null;
        for (EnableStatus status : values()) {
            if (status.value == value) {
                return status;
            }
        }
        return null;
    }
}
