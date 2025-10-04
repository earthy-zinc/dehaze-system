package com.pei.dehaze.module.iot.enums.rule;

import com.pei.dehaze.framework.common.core.ArrayValuable;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.Arrays;

/**
 * IoT 告警配置的接收方式枚举
 *
 * @author earthyzinc
 */
@RequiredArgsConstructor
@Getter
public enum IotAlertConfigReceiveTypeEnum implements ArrayValuable<Integer> {

    SMS(1), // 短信
    MAIL(2), // 邮箱
    NOTIFY(3); // 通知

    private final Integer type;

    public static final Integer[] ARRAYS = Arrays.stream(values()).map(IotAlertConfigReceiveTypeEnum::getType).toArray(Integer[]::new);

    @Override
    public Integer[] array() {
        return ARRAYS;
    }

}
