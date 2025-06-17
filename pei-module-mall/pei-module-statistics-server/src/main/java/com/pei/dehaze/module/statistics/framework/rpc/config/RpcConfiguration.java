package com.pei.dehaze.module.statistics.framework.rpc.config;

import com.pei.dehaze.module.product.api.spu.ProductSpuApi;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.context.annotation.Configuration;

@Configuration(value = "statisticsRpcConfiguration", proxyBeanMethods = false)
@EnableFeignClients(clients = {ProductSpuApi.class})
public class RpcConfiguration {
}
