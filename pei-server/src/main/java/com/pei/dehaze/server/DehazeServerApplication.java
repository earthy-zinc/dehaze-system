package com.pei.dehaze.server;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 项目的启动类
 * @author earthyzinc
 */
@SpringBootApplication(scanBasePackages = {"${pei.info.base-package}.server", "${pei.info.base-package}.module"},
        excludeName = {
            // RPC 相关
//            "org.springframework.cloud.openfeign.FeignAutoConfiguration",
//            "com.pei.dehaze.module.system.framework.rpc.config.RpcConfiguration"
        })
public class DehazeServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(DehazeServerApplication.class, args);
//        new SpringApplicationBuilder(DehazeServerApplication.class)
//                .applicationStartup(new BufferingApplicationStartup(20480))
//                .run(args);
    }
}
