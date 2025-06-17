package com.pei.dehaze.module.iot.framework.quartz;

import com.baomidou.dynamic.datasource.annotation.DS;

import java.lang.annotation.*;

/**
 * @author earthy-zinc
 * @since 2025-06-11 18:04:24
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@DS("quartz")
public @interface QuartzDS {
}
