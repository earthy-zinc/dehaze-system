package com.pei.dehaze;

import com.pei.dehaze.common.util.LogArchiveUtils;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class SystemApplication {
    public static void main(String[] args) {
        // dev 环境启动时归档当天旧日志，使本次启动日志写入全新活动文件；
        // 必须在 SpringApplication.run 触发 logback 打开文件前执行。生产环境(SPRING_PROFILES_ACTIVE=prod)不归档。
        String profile = System.getenv("SPRING_PROFILES_ACTIVE");
        if (profile == null || "dev".equals(profile)) {
            LogArchiveUtils.archiveTodayLogs();
        }
        SpringApplication.run(SystemApplication.class, args);
    }
}
