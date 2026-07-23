package com.pei.dehaze.sdk.model.user;

/**
 * 性别枚举
 * 对齐后端 User.gender：0=未知，1=男，2=女
 */
public enum Gender {
    UNKNOWN(0, "未知"),
    MALE(1, "男"),
    FEMALE(2, "女");

    private final int value;
    private final String label;

    Gender(int value, String label) {
        this.value = value;
        this.label = label;
    }

    public int getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    public static Gender fromValue(Integer value) {
        if (value == null) return null;
        for (Gender gender : values()) {
            if (gender.value == value) {
                return gender;
            }
        }
        return null;
    }
}
