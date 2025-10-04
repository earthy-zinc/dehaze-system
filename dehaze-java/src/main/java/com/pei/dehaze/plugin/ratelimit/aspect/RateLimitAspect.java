package com.pei.dehaze.plugin.ratelimit.aspect;

import com.pei.dehaze.common.exception.RateLimitException;
import com.pei.dehaze.plugin.ratelimit.annotation.RateLimit;
import com.pei.dehaze.security.util.SecurityUtils;
import jakarta.servlet.http.HttpServletRequest;
import jodd.util.StringUtil;
import lombok.RequiredArgsConstructor;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.redisson.api.RRateLimiter;
import org.redisson.api.RateIntervalUnit;
import org.redisson.api.RateType;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Component;

import java.lang.reflect.Method;

@Aspect
@Component
@RequiredArgsConstructor
public class RateLimitAspect {
    private final RedissonClient redissonClient;
    private final HttpServletRequest request;

    @Around("@annotation(rateLimit)")
    public Object around(ProceedingJoinPoint joinPoint, RateLimit rateLimit) throws Throwable {
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();

        // 构造限流key
        String key = buildRateLimitKey(rateLimit, method);

        // 获取限流器
        RRateLimiter rateLimiter = redissonClient.getRateLimiter(key);

        // 初始化限流规则（只在首次设置）
        if (!rateLimiter.isExists()) {
            rateLimiter.trySetRate(
                    RateType.OVERALL,
                    rateLimit.maxRequests(),
                    rateLimit.timeWindow(),
                    RateIntervalUnit.SECONDS
            );
        }

        // 尝试获取令牌
        if (!rateLimiter.tryAcquire()) {
            throw new RateLimitException(rateLimit.message());
        }

        return joinPoint.proceed();
    }

    private String buildRateLimitKey(RateLimit rateLimit, Method method) {
        StringBuilder key = new StringBuilder(rateLimit.key());

        switch (rateLimit.type()) {
            case IP:
                key.append(getClientIp());
                break;
            case USER:
                key.append(SecurityUtils.getUserId());
                break;
            case GLOBAL:
                key.append("global");
                break;
        }

        key.append(":")
                .append(method.getDeclaringClass().getName())
                .append("#")
                .append(method.getName());

        return key.toString();
    }

    private String getClientIp() {
        String ip = request.getHeader("X-Forwarded-For");
        if (StringUtil.isEmpty(ip) || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getRemoteAddr();
        }
        return ip;
    }
}
