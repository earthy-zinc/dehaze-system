package com.pei.dehaze.config;

import cn.hutool.core.collection.CollUtil;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.config.property.SecurityProperties;
import com.pei.dehaze.filter.JwtValidationFilter;
import com.pei.dehaze.security.exception.MyAccessDeniedHandler;
import com.pei.dehaze.security.exception.MyAuthenticationEntryPoint;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityCustomizer;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.annotation.web.configurers.HeadersConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

/**
 * Spring Security 权限配置
 *
 * @author Ray Hao
 * @since 2023/2/17
 */
@Slf4j
@Configuration
@EnableWebSecurity
@EnableMethodSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private static final String DEFAULT_DEV_KEY = "SecretKey012345678901234567890123456789012345678901234567890123456789";

    private final MyAuthenticationEntryPoint authenticationEntryPoint;
    private final MyAccessDeniedHandler accessDeniedHandler;
    private final RedisTemplate<String, Object> redisTemplate;
    private final SecurityProperties securityProperties;

    @PostConstruct
    public void validateJwtKey() {
        String key = securityProperties.getJwt().getKey();
        if (DEFAULT_DEV_KEY.equals(key)) {
            log.warn("⚠️  安全警告：当前使用硬编码的默认 JWT 密钥！请设置环境变量 JWT_SECRET_KEY 以使用安全密钥。" +
                    "生产环境必须设置，否则攻击者可伪造任意用户 Token。");
        }
    }



    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                .authorizeHttpRequests(requestMatcherRegistry ->
                        requestMatcherRegistry
                                .requestMatchers(SecurityConstants.LOGIN_PATH).permitAll()
                                .requestMatchers("/health", "/ready").permitAll()
                                .requestMatchers("/actuator/health").permitAll()
                                .requestMatchers("/actuator/**").hasRole("ADMIN")
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

        // JWT 校验过滤器
        http.addFilterBefore(new JwtValidationFilter(redisTemplate,securityProperties.getJwt().getKey()), UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }

    /**
     * 不走过滤器链的放行配置
     */
    @Bean
    public WebSecurityCustomizer webSecurityCustomizer() {
        return web -> {
            if (CollUtil.isNotEmpty(securityProperties.getIgnoreUrls())) {
                web.ignoring().requestMatchers(securityProperties.getIgnoreUrls().toArray(new String[0]));
            }
        };
    }

    /**
     * 密码编码器
     */
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    /**
     * AuthenticationManager 手动注入
     *
     * @param authenticationConfiguration 认证配置
     */
    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration authenticationConfiguration) throws Exception {
        return authenticationConfiguration.getAuthenticationManager();
    }

}
