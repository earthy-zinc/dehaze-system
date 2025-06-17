package com.pei.dehaze.framework.dict.config;

import com.pei.dehaze.framework.common.biz.system.dict.DictDataCommonApi;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.cloud.openfeign.EnableFeignClients;

/**
 * 字典用到 Feign 的配置项
 *
 * @author earthyzinc
 */
@AutoConfiguration
@EnableFeignClients(clients = DictDataCommonApi.class) // 主要是引入相关的 API 服务
public class PeiDictRpcAutoConfiguration {
}
