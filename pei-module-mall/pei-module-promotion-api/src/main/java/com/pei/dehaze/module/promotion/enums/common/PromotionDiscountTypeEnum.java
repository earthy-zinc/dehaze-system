package com.pei.dehaze.module.promotion.enums.common;

import com.pei.dehaze.framework.common.core.ArrayValuable;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Arrays;

/**
 * 优惠类型枚举
 *
 * @author earthyzinc
 */
@Getter
@AllArgsConstructor
public enum PromotionDiscountTypeEnum implements ArrayValuable<Integer> {

    PRICE(1, "满减"), // 具体金额
    PERCENT(2, "折扣"), // 百分比
    ;

    public static final Integer[] ARRAYS = Arrays.stream(values()).map(PromotionDiscountTypeEnum::getType).toArray(Integer[]::new);

    /**
     * 优惠类型
     */
    private final Integer type;
    /**
     * 名字
     */
    private final String name;

    @Override
    public Integer[] array() {
        return ARRAYS;
    }

}
