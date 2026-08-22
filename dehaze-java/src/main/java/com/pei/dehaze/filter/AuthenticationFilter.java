package com.pei.dehaze.filter;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.ResponseUtils;
import com.pei.dehaze.security.model.SysUserDetails;
import com.pei.dehaze.service.ApiKeyService;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.MDC;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpHeaders;
import org.springframework.lang.NonNull;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * 统一认证过滤器：根据请求凭证类型二选一。
 * Bearer dhak_* -> API Key 认证；Session Cookie/Header -> Session 认证。
 */
public class AuthenticationFilter extends OncePerRequestFilter {

    private static final String API_KEY_PREFIX = "dhak_";
    private static final String BEARER_PREFIX = "Bearer ";

    private final ApiKeyService apiKeyService;
    private final StringRedisTemplate redisTemplate;

    public AuthenticationFilter(ApiKeyService apiKeyService, StringRedisTemplate redisTemplate) {
        this.apiKeyService = apiKeyService;
        this.redisTemplate = redisTemplate;
    }

    @Override
    protected void doFilterInternal(
        @NonNull HttpServletRequest request,
        @NonNull HttpServletResponse response, 
        @NonNull FilterChain chain
    )
            throws ServletException, IOException {

        String authHeader = request.getHeader(HttpHeaders.AUTHORIZATION);

        if (authHeader != null && authHeader.startsWith(BEARER_PREFIX)) {
            String token = authHeader.substring(BEARER_PREFIX.length()).trim();
            if (token.startsWith(API_KEY_PREFIX)) {
                authenticateWithApiKey(token, response, chain, request);
                return;
            }
        }

        authenticateWithSession(request, response, chain);
    }

    private void authenticateWithApiKey(
            String apiKey, HttpServletResponse response, FilterChain chain,
            HttpServletRequest request)
            throws ServletException, IOException {
        Authentication authentication = apiKeyService.authenticateByKey(apiKey);
        if (authentication == null) {
            ResponseUtils.writeErrMsg(response, ResultCode.TOKEN_INVALID);
            return;
        }

        SecurityContextHolder.getContext().setAuthentication(authentication);

        Object principal = authentication.getPrincipal();
        if (principal instanceof SysUserDetails userDetails) {
            if (userDetails.getUserId() != null) {
                MDC.put("user_id", userDetails.getUserId().toString());
            }
        }

        try {
            chain.doFilter(request, response);
        } finally {
            MDC.remove("user_id");
            SecurityContextHolder.clearContext();
        }
    }

    private void authenticateWithSession(
            HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {

        String sessionId = extractSessionId(request);
        if (sessionId == null) {
            ResponseUtils.writeErrMsg(response, ResultCode.TOKEN_INVALID);
            return;
        }

        String sessionJson = redisTemplate.opsForValue().get(SecurityConstants.SESSION_PREFIX + sessionId);
        if (sessionJson == null) {
            ResponseUtils.writeErrMsg(response, ResultCode.TOKEN_INVALID);
            return;
        }

        JSONObject session = JSONUtil.parseObj(sessionJson);

        SysUserDetails userDetails = new SysUserDetails();
        userDetails.setUserId(session.getLong("userId"));
        userDetails.setUsername(session.getStr("username"));
        userDetails.setDeptId(session.getLong("deptId"));
        userDetails.setDataScope(session.getInt("dataScope"));
        userDetails.setNickname(session.getStr("nickname"));

        JSONArray authoritiesArray = session.getJSONArray("authorities");
        Set<SimpleGrantedAuthority> authorities = authoritiesArray.stream()
                .map(Object::toString)
                .map(SimpleGrantedAuthority::new)
                .collect(Collectors.toSet());
        userDetails.setAuthorities(authorities);

        UsernamePasswordAuthenticationToken authentication =
                new UsernamePasswordAuthenticationToken(userDetails, "", authorities);
        SecurityContextHolder.getContext().setAuthentication(authentication);

        if (userDetails.getUserId() != null) {
            MDC.put("user_id", userDetails.getUserId().toString());
        }

        try {
            Long ttl = redisTemplate.getExpire(SecurityConstants.SESSION_PREFIX + sessionId, TimeUnit.SECONDS);
            if (ttl != null && ttl > 0 && ttl < SecurityConstants.RENEW_THRESHOLD) {
                redisTemplate.expire(SecurityConstants.SESSION_PREFIX + sessionId,
                        SecurityConstants.SESSION_TTL, TimeUnit.SECONDS);
            }

            chain.doFilter(request, response);
        } finally {
            MDC.remove("user_id");
            SecurityContextHolder.clearContext();
        }
    }

    private String extractSessionId(HttpServletRequest request) {
        Cookie[] cookies = request.getCookies();
        if (cookies != null) {
            for (Cookie cookie : cookies) {
                if (SecurityConstants.SESSION_COOKIE_NAME.equals(cookie.getName())) {
                    return cookie.getValue();
                }
            }
        }
        return request.getHeader(SecurityConstants.SESSION_COOKIE_NAME);
    }

    @Override
    protected boolean shouldNotFilter(@NonNull HttpServletRequest request) {
        String path = request.getRequestURI();
        return path.equals(SecurityConstants.LOGIN_PATH)
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
