package com.pei.dehaze.config;

import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.filter.AuthenticationFilter;
import com.pei.dehaze.security.exception.MyAccessDeniedHandler;
import com.pei.dehaze.security.exception.MyAuthenticationEntryPoint;
import com.pei.dehaze.service.ApiKeyService;
import com.pei.dehaze.service.client.AiProxyRoutes;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.annotation.web.configurers.HeadersConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

@Configuration
@EnableWebSecurity
@EnableMethodSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final MyAuthenticationEntryPoint authenticationEntryPoint;
    private final MyAccessDeniedHandler accessDeniedHandler;
    private final StringRedisTemplate stringRedisTemplate;
    private final ApiKeyService apiKeyService;

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                .authorizeHttpRequests(requestMatcherRegistry -> {
                    // 第三方协议端点（OpenAI/Claude 兼容、A2A）免本地鉴权：凭据形态由 python 终审，
                    // 清单与 AuthenticationFilter 豁免同源（AiProxyRoutes）
                    AiProxyRoutes.authExemptRoutes().forEach(route ->
                            requestMatcherRegistry
                                    .requestMatchers(HttpMethod.valueOf(route.method()), route.pattern())
                                    .permitAll());
                    requestMatcherRegistry
                            .requestMatchers(SecurityConstants.LOGIN_PATH).permitAll()
                            .requestMatchers("/health", "/ready").permitAll()
                            .requestMatchers("/actuator/health").permitAll()
                            .requestMatchers("/actuator/**").hasRole("ADMIN")
                            .requestMatchers("/v3/api-docs/**").permitAll()
                            .requestMatchers("/doc.html").permitAll()
                            .requestMatchers("/swagger-resources/**").permitAll()
                            .requestMatchers("/webjars/**").permitAll()
                            .requestMatchers("/swagger-ui/**").permitAll()
                            .requestMatchers("/swagger-ui.html").permitAll()
                            // 文件下载不再公开：登录 + 归属校验（B0407），对齐 Python/Go 端
                            .requestMatchers("/api/v1/auth/register").permitAll()
                            .requestMatchers("/api/v1/auth/captcha").permitAll()
                            .requestMatchers("/api/v1/orders/payment/wechat/callback").permitAll()
                            .requestMatchers("/api/v1/orders/payment/alipay/callback").permitAll()
                            .requestMatchers("/api/v1/logs/client").permitAll()
                            // 语音流式 ASR WebSocket：浏览器握手无法携带自定义头，会话凭证随 query 的
                            // sid 传递，鉴权与 ASR 会话归属由 python 终审（对齐 Go 端不挂鉴权中间件）
                            .requestMatchers("/ws/asr").permitAll()
                            .anyRequest().authenticated();
                })
                .exceptionHandling(httpSecurityExceptionHandlingConfigurer ->
                        httpSecurityExceptionHandlingConfigurer
                                .authenticationEntryPoint(authenticationEntryPoint)
                                .accessDeniedHandler(accessDeniedHandler)
                )
                .sessionManagement(configurer -> configurer.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .csrf(AbstractHttpConfigurer::disable)
                .headers(headers -> headers.frameOptions(HeadersConfigurer.FrameOptionsConfig::disable))
        ;

        http.addFilterBefore(new AuthenticationFilter(apiKeyService, stringRedisTemplate), UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration authenticationConfiguration) throws Exception {
        return authenticationConfiguration.getAuthenticationManager();
    }

}
