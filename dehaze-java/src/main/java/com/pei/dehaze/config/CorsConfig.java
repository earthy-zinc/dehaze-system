package com.pei.dehaze.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;

import java.util.List;

/**
 * CORS 跨域配置
 * <p>
 * 从 application-{profile}.yml 的 cors.allowed-origins 读取 Origin 白名单，
 * 配合 allowCredentials=true 使用合法的组合（禁止 "*" + credentials）。
 *
 * @author earthyzinc
 * @since 2023/4/17
 */
@Configuration
@ConfigurationProperties(prefix = "cors")
@Data
public class CorsConfig {

    /**
     * 允许的 Origin 白名单（从 yml 配置读取）
     */
    private List<String> allowedOrigins;

    @Bean
    public FilterRegistrationBean<CorsFilter> filterRegistrationBean() {
        CorsConfiguration corsConfiguration = new CorsConfiguration();
        // 1. 仅允许白名单内的 Origin（禁止 "*" + credentials 组合）
        corsConfiguration.setAllowedOrigins(allowedOrigins);
        // 2. 允许任何请求头
        corsConfiguration.addAllowedHeader(CorsConfiguration.ALL);
        // 3. 允许任何方法
        corsConfiguration.addAllowedMethod(CorsConfiguration.ALL);
        // 4. 允许凭证（与白名单 Origin 组合合法）
        corsConfiguration.setAllowCredentials(true);
        // 5. 暴露 TraceId 等自定义响应头
        corsConfiguration.addExposedHeader("X-Trace-Id");

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", corsConfiguration);
        CorsFilter corsFilter = new CorsFilter(source);

        FilterRegistrationBean<CorsFilter> filterRegistrationBean = new FilterRegistrationBean<>(corsFilter);
        filterRegistrationBean.setOrder(-101);  // 小于 SpringSecurity Filter的 Order(-100) 即可

        return filterRegistrationBean;
    }
}
