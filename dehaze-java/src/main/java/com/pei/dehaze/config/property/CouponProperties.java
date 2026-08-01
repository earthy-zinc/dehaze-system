package com.pei.dehaze.config.property;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * 优惠券相关配置
 */
@Data
@Component
@ConfigurationProperties(prefix = "coupon")
public class CouponProperties {

    /**
     * 领取限流次数（每分钟）
     */
    private Integer receiveRateLimit = 5;

    /**
     * 领取限流时间窗口（秒）
     */
    private Integer receiveRateWindow = 60;
}
