package com.pei.dehaze.common.enums;

import com.baomidou.mybatisplus.annotation.EnumValue;
import com.fasterxml.jackson.annotation.JsonValue;
import com.pei.dehaze.common.base.IBaseEnum;
import lombok.Getter;

@Getter
public enum LogStatusEnum implements IBaseEnum<Integer> {

    PROCESSING(1, "处理中"),
    COMPLETED(2, "已完成"),
    FAILED(3, "失败");

    @JsonValue
    @EnumValue
    private final Integer value;

    private final String label;

    LogStatusEnum(Integer value, String label) {
        this.value = value;
        this.label = label;
    }
}
