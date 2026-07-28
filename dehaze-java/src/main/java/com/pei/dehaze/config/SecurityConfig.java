package com.pei.dehaze.config;

import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.filter.SessionFilter;
import com.pei.dehaze.security.exception.MyAccessDeniedHandler;
import com.pei.dehaze.security.exception.MyAuthenticationEntryPoint;
import com.pei.dehaze.service.ApiKeyService;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.StringRedisTemplate;
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
                .authorizeHttpRequests(requestMatcherRegistry ->
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
                                .requestMatchers("/api/v1/files/download/**").permitAll()
                                .requestMatchers("/api/v1/auth/register").permitAll()
                                .requestMatchers("/api/v1/auth/captcha").permitAll()
                                .requestMatchers("/api/v1/payments/wechat/callback").permitAll()
                                .requestMatchers("/api/v1/payments/alipay/callback").permitAll()
                                .anyRequest().authenticated()
                )
                .exceptionHandling(httpSecurityExceptionHandlingConfigurer ->
                        httpSecurityExceptionHandlingConfigurer
                                .authenticationEntryPoint(authenticationEntryPoint)
                                .accessDeniedHandler(accessDeniedHandler)
                )
                .sessionManagement(configurer -> configurer.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .csrf(AbstractHttpConfigurer::disable)
                .headers(headers -> headers.frameOptions(HeadersConfigurer.FrameOptionsConfig::disable))
        ;

        http.addFilterBefore(new SessionFilter(stringRedisTemplate, apiKeyService), UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration authenticationConfiguration) throws Exception {
        return authenticationConfiguration.getAuthenticationManager();
    }

}
