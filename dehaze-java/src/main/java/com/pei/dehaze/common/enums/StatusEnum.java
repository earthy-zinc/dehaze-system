package com.pei.dehaze.common.enums;

import com.baomidou.mybatisplus.annotation.EnumValue;
import com.fasterxml.jackson.annotation.JsonValue;
import com.pei.dehaze.common.base.IBaseEnum;
import lombok.Getter;

/**
 * 状态枚举
 *
 * @author earthyzinc
 * @since 2022/10/14
 */
@Getter
public enum StatusEnum implements IBaseEnum<Integer> {

    ENABLE(1, "启用"),
    DISABLE (0, "禁用");

    @JsonValue
    @EnumValue
    private final Integer value;

    private final String label;

    StatusEnum(Integer value, String label) {
        this.value = value;
        this.label = label;
    }
}
