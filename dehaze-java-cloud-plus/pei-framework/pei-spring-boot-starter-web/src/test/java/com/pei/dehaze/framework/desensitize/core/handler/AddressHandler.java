package com.pei.dehaze.framework.desensitize.core.handler;

import com.pei.dehaze.framework.desensitize.core.DesensitizeTest;
import com.pei.dehaze.framework.desensitize.core.annotation.Address;
import com.pei.dehaze.framework.desensitize.core.base.handler.DesensitizationHandler;

/**
 * {@link Address} 的脱敏处理器
 * <p>
 * 用于 {@link DesensitizeTest} 测试使用
 */
public class AddressHandler implements DesensitizationHandler<Address> {

    @Override
    public String desensitize(String origin, Address annotation) {
        return origin + annotation.replacer();
    }

}
