package com.pei.dehaze.config;

import com.pei.dehaze.config.property.SecurityProperties;
import com.pei.dehaze.security.exception.MyAccessDeniedHandler;
import com.pei.dehaze.security.exception.MyAuthenticationEntryPoint;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.FilterType;
import org.springframework.context.annotation.Primary;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;

/**
 * 统一测试配置类
 * <p>
 * 用于集成测试和WebMvcTest场景，提供简化的Security配置
 * 排除Security和Redis等可能导致问题的自动配置
 */
@TestConfiguration
@SpringBootApplication(
        scanBasePackages = "com.pei.dehaze",
        exclude = {
                SecurityAutoConfiguration.class,
                RedisAutoConfiguration.class
        }
)
@ComponentScan(
        basePackages = "com.pei.dehaze",
        excludeFilters = {
                @ComponentScan.Filter(type = FilterType.REGEX, pattern = "com\\.pei\\.dehaze\\.security\\..*"),
                @ComponentScan.Filter(type = FilterType.ASSIGNABLE_TYPE, classes = {
                        com.pei.dehaze.config.SecurityConfig.class
                })
        }
)
public class TestConfig {

    @Bean
    @Primary
    public SecurityFilterChain testSecurityFilterChain(HttpSecurity http) throws Exception {
        http
                .authorizeHttpRequests(auth -> auth
                        .anyRequest().authenticated()
                )
                .sessionManagement(session -> session
                        .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                )
                .csrf(AbstractHttpConfigurer::disable)
                .exceptionHandling(exception -> exception
                        .authenticationEntryPoint((request, response, authException) -> {
                            response.setStatus(401);
                            response.setContentType("application/json;charset=UTF-8");
                            response.getWriter().write("{\"code\":\"A0301\",\"msg\":\"未授权\"}");
                        })
                        .accessDeniedHandler((request, response, accessDeniedException) -> {
                            response.setStatus(403);
                            response.setContentType("application/json;charset=UTF-8");
                            response.getWriter().write("{\"code\":\"A0301\",\"msg\":\"访问被拒绝\"}");
                        })
                );
        return http.build();
    }

    @Bean
    public MyAuthenticationEntryPoint myAuthenticationEntryPoint() {
        return new MyAuthenticationEntryPoint();
    }

    @Bean
    public MyAccessDeniedHandler myAccessDeniedHandler() {
        return new MyAccessDeniedHandler();
    }

    @Bean
    public SecurityProperties securityProperties() {
        return new SecurityProperties();
    }
}
