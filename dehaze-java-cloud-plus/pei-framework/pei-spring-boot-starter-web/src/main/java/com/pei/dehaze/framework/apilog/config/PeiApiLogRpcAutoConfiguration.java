package com.pei.dehaze.framework.apilog.config;

import com.pei.dehaze.framework.common.biz.infra.logger.ApiAccessLogCommonApi;
import com.pei.dehaze.framework.common.biz.infra.logger.ApiErrorLogCommonApi;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.cloud.openfeign.EnableFeignClients;

/**
 * API 日志使用到 Feign 的配置项
 *
 * @author earthyzinc
 */
@AutoConfiguration
@EnableFeignClients(clients = {ApiAccessLogCommonApi.class, ApiErrorLogCommonApi.class}) // 主要是引入相关的 API 服务
public class PeiApiLogRpcAutoConfiguration {
}
