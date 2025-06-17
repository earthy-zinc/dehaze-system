package com.pei.dehaze.module.mp.framework.rpc.config;

import com.pei.dehaze.module.infra.api.file.FileApi;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.context.annotation.Configuration;

@Configuration(value = "mpRpcConfiguration", proxyBeanMethods = false)
@EnableFeignClients(clients = FileApi.class)
public class RpcConfiguration {
}
