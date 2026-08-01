package com.pei.dehaze.filter;

import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.ResponseUtils;
import com.pei.dehaze.service.ApiKeyService;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.MDC;
import org.springframework.http.HttpHeaders;
import org.springframework.lang.NonNull;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

/**
 * API Key 认证过滤器 — 独立负责 Bearer dhak_* 形式的 API Key 认证。
 * 与 {@link SessionFilter} 解耦，遵循单一职责原则。
 * <p>
 * 优先级应在 SessionFilter 之前，拦截所有需要认证的请求。
 */
public class ApiKeyAuthenticationFilter extends OncePerRequestFilter {

    private static final String API_KEY_PREFIX = "dhak_";
    private static final String BEARER_PREFIX = "Bearer ";

    private final ApiKeyService apiKeyService;

    public ApiKeyAuthenticationFilter(ApiKeyService apiKeyService) {
        this.apiKeyService = apiKeyService;
    }

    @Override
    protected void doFilterInternal(
            @NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {

        String authHeader = request.getHeader(HttpHeaders.AUTHORIZATION);
        if (authHeader == null) {
            // non-API-Key 请求（如 Session Cookie），透传给下一个 Filter
            chain.doFilter(request, response);
            return;
        }

        if (authHeader.startsWith(BEARER_PREFIX)) {
            authHeader = authHeader.substring(BEARER_PREFIX.length());
        }

        if (!authHeader.startsWith(API_KEY_PREFIX)) {
            // 非 dhak_ 前缀，可能是其他认证方式，透传
            chain.doFilter(request, response);
            return;
        }

        Authentication authentication = apiKeyService.authenticateByKey(authHeader);
        if (authentication == null) {
            ResponseUtils.writeErrMsg(response, ResultCode.TOKEN_INVALID);
            return;
        }

        SecurityContextHolder.getContext().setAuthentication(authentication);

        // 认证通过后，将 user_id 写入 MDC（供日志自动注入到每条日志）
        Object principal = authentication.getPrincipal();
        String userIdStr = null;
        if (principal instanceof com.pei.dehaze.security.model.SysUserDetails) {
            Long userId = ((com.pei.dehaze.security.model.SysUserDetails) principal).getUserId();
            if (userId != null) {
                userIdStr = userId.toString();
            }
        }
        if (userIdStr != null) {
            MDC.put("user_id", userIdStr);
        }

        try {
            chain.doFilter(request, response);
        } finally {
            MDC.remove("user_id");
        }
    }

    @Override
    protected boolean shouldNotFilter(@NonNull HttpServletRequest request) {
        String path = request.getRequestURI();
        return path.equals("/api/v1/auth/login")
                || path.equals("/api/v1/auth/register")
                || path.equals("/api/v1/auth/captcha")
                || path.startsWith("/health")
                || path.startsWith("/ready")
                || path.startsWith("/actuator")
                || path.startsWith("/v3/api-docs")
                || path.equals("/doc.html")
                || path.startsWith("/swagger-ui")
                || path.startsWith("/webjars")
                || path.startsWith("/api/v1/files/download");
    }
}
