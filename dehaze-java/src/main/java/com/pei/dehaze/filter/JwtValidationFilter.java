package com.pei.dehaze.filter;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.jwt.JWT;
import cn.hutool.jwt.JWTException;
import cn.hutool.jwt.JWTUtil;
import cn.hutool.jwt.RegisteredPayload;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.ResponseUtils;
import com.pei.dehaze.security.util.JwtUtils;
import com.pei.dehaze.service.ApiKeyService;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.HttpHeaders;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

public class JwtValidationFilter extends OncePerRequestFilter {

    private final RedisTemplate<String, Object> redisTemplate;

    private final byte[] secretKey;

    private final ApiKeyService apiKeyService;

    public JwtValidationFilter(RedisTemplate<String, Object> redisTemplate, String secretKey, ApiKeyService apiKeyService) {
        this.redisTemplate = redisTemplate;
        this.secretKey = secretKey.getBytes();
        this.apiKeyService = apiKeyService;
    }


    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        String token = request.getHeader(HttpHeaders.AUTHORIZATION);

        try {
            if (CharSequenceUtil.isNotBlank(token) && token.startsWith(SecurityConstants.JWT_TOKEN_PREFIX)) {
                token = token.substring(SecurityConstants.JWT_TOKEN_PREFIX.length());

                if (token.startsWith("dhak_")) {
                    Authentication authentication = apiKeyService.authenticateByKey(token);
                    if (authentication != null) {
                        SecurityContextHolder.getContext().setAuthentication(authentication);
                    } else {
                        ResponseUtils.writeErrMsg(response, ResultCode.TOKEN_INVALID);
                        return;
                    }
                    filterChain.doFilter(request, response);
                    return;
                }

                if (JWTUtil.verify(token, secretKey)) {
                    JWT jwt = JWTUtil.parseToken(token);
                    JSONObject payloads = jwt.getPayloads();

                    Long expiresAt = payloads.getLong(RegisteredPayload.EXPIRES_AT);
                    if (expiresAt != null) {
                        long currentTimeSeconds = System.currentTimeMillis() / 1000;
                        if (expiresAt < currentTimeSeconds) {
                            ResponseUtils.writeErrMsg(response, ResultCode.TOKEN_INVALID);
                            return;
                        }
                    }

                    String jti = payloads.getStr(RegisteredPayload.JWT_ID);
                    boolean isTokenBlacklisted = Boolean.TRUE.equals(redisTemplate.hasKey(SecurityConstants.BLACKLIST_TOKEN_PREFIX + jti));
                    if (isTokenBlacklisted) {
                        ResponseUtils.writeErrMsg(response, ResultCode.TOKEN_INVALID);
                        return;
                    }
                    Authentication authentication = JwtUtils.getAuthentication(payloads);
                    SecurityContextHolder.getContext().setAuthentication(authentication);
                } else {
                    ResponseUtils.writeErrMsg(response, ResultCode.TOKEN_INVALID);
                    return;
                }
            }
        } catch (JWTException e) {
            SecurityContextHolder.clearContext();
            ResponseUtils.writeErrMsg(response, ResultCode.TOKEN_INVALID);
            return;
        }
        filterChain.doFilter(request, response);
    }
}
