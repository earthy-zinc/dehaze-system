package com.pei.dehaze.common.enums;

import com.baomidou.mybatisplus.annotation.EnumValue;
import com.fasterxml.jackson.annotation.JsonValue;
import com.pei.dehaze.common.base.IBaseEnum;
import lombok.Getter;

import java.util.Set;

@Getter
public enum TaskStatusEnum implements IBaseEnum<Integer> {

    PENDING(1, "待处理"),
    PROCESSING(2, "处理中"),
    COMPLETED(3, "已完成"),
    FAILED(4, "失败"),
    CANCELLED(5, "已取消");

    @JsonValue
    @EnumValue
    private final Integer value;

    private final String label;

    TaskStatusEnum(Integer value, String label) {
        this.value = value;
        this.label = label;
    }

    public static final Set<Integer> TERMINAL_STATUSES = Set.of(COMPLETED.value, FAILED.value, CANCELLED.value);
}
