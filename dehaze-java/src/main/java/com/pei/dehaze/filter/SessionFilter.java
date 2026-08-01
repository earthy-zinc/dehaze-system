package com.pei.dehaze.filter;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.security.model.SysUserDetails;
import com.pei.dehaze.common.util.ResponseUtils;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.MDC;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.lang.NonNull;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * Session 认证过滤器 — 纯 Session Cookie/Header 认证，不处理 API Key。
 * API Key 认证由 {@link ApiKeyAuthenticationFilter} 独立负责。
 */
public class SessionFilter extends OncePerRequestFilter {

    private final StringRedisTemplate redisTemplate;

    public SessionFilter(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    @Override
    protected void doFilterInternal(
            @NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response, FilterChain chain)
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

        // 认证通过后，将 user_id 写入 MDC（供日志自动注入到每条日志）
        Long userId = userDetails.getUserId();
        if (userId != null) {
            MDC.put("user_id", userId.toString());
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
