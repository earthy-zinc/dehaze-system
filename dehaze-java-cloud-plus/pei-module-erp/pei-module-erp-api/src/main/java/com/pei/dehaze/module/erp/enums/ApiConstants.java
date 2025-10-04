package com.pei.dehaze.module.erp.enums;

import com.pei.dehaze.framework.common.enums.RpcConstants;

/**
 * API 相关的枚举
 *
 * @author earthyzinc
 */
public class ApiConstants {

    /**
     * 服务名
     *
     * 注意，需要保证和 spring.application.name 保持一致
     */
    public static final String NAME = "erp-server";

    public static final String PREFIX = RpcConstants.RPC_API_PREFIX + "/erp";

    public static final String VERSION = "1.0.0";

}
