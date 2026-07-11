package com.pei.dehaze.config;

import com.pei.dehaze.config.property.AlgorithmProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;

/**
 * HTTP 客户端配置 —— 用于调用 Python 算法服务
 * <p>
 * 使用 Spring Boot 内置的 RestTemplateBuilder，无需额外 HTTP 客户端依赖。
 * 超时通过 Duration 配置，连接池由底层 JDK HttpURLConnection 管理。
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Slf4j
@Configuration
@RequiredArgsConstructor
public class RestClientConfig {

    private final AlgorithmProperties algorithmProperties;

    /**
     * 算法服务专用 RestTemplate
     */
    @Bean(name = "algorithmRestTemplate")
    public RestTemplate algorithmRestTemplate(RestTemplateBuilder builder) {
        return builder
                .setConnectTimeout(Duration.ofMillis(algorithmProperties.getConnectTimeout()))
                .setReadTimeout(Duration.ofMillis(algorithmProperties.getReadTimeout()))
                .build();
    }
}
