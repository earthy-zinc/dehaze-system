package com.pei.dehaze.plugin.ratelimit.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface RateLimit {
    /**
     * 限流key前缀
     */
    String key() default "rate_limit:";

    /**
     * 时间窗口（秒）
     */
    int timeWindow() default 60;

    /**
     * 窗口期内最大请求数
     */
    int maxRequests() default 100;

    /**
     * 限流类型
     */
    LimitType type() default LimitType.IP;

    /**
     * 限流提示信息
     */
    String message() default "请求过于频繁，请稍后再试";

    /**
     * 限流器类型
     */
    LimiterType limiter() default LimiterType.TOKEN_BUCKET;

    enum LimitType {
        IP,      // 按客户端IP限流
        USER,    // 按登录用户限流
        GLOBAL   // 全局限流
    }

    enum LimiterType {
        TOKEN_BUCKET,  // 令牌桶算法
        FIXED_WINDOW   // 固定窗口算法
    }
}
