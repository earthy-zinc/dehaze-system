package com.pei.dehaze.plugin.ratelimit.aspect;

import cn.hutool.core.text.CharSequenceUtil;
import com.pei.dehaze.common.exception.RateLimitException;
import com.pei.dehaze.plugin.ratelimit.annotation.RateLimit;
import com.pei.dehaze.security.util.SecurityUtils;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.redisson.api.RAtomicLong;
import org.redisson.api.RRateLimiter;
import org.redisson.api.RateIntervalUnit;
import org.redisson.api.RateType;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Component;

import java.util.concurrent.TimeUnit;

@Aspect
@Component
@RequiredArgsConstructor
public class RateLimitAspect {
    private final RedissonClient redissonClient;
    private final HttpServletRequest request;

    private static final String[] IP_HEADERS = {
            "X-Forwarded-For",
            "X-Real-IP",
            "Proxy-Client-IP",
            "WL-Proxy-Client-IP"
    };

    @Around("@annotation(rateLimit)")
    public Object around(ProceedingJoinPoint joinPoint, RateLimit rateLimit) throws Throwable {
        // 构造限流key：{prefix}{ip|userId|global}
        // 三端统一前缀为 rate:limit:，与 Python rate:limit:{path}:{ip} 共享命名空间
        String key = buildRateLimitKey(rateLimit);

        if (rateLimit.limiter() == RateLimit.LimiterType.FIXED_WINDOW) {
            // 固定窗口：INCR + EXPIRE，与 Python rate_limit.py 对齐
            RAtomicLong counter = redissonClient.getAtomicLong(key);
            long count = counter.incrementAndGet();
            if (count == 1) {
                counter.expire(rateLimit.timeWindow(), TimeUnit.SECONDS);
            }
            if (count > rateLimit.maxRequests()) {
                throw new RateLimitException(rateLimit.message());
            }
        } else {
            // 令牌桶：Redisson RRateLimiter（GCRA）
            RRateLimiter rateLimiter = redissonClient.getRateLimiter(key);
            if (!rateLimiter.isExists()) {
                rateLimiter.trySetRate(
                        RateType.OVERALL,
                        rateLimit.maxRequests(),
                        rateLimit.timeWindow(),
                        RateIntervalUnit.SECONDS
                );
            }
            if (!rateLimiter.tryAcquire()) {
                throw new RateLimitException(rateLimit.message());
            }
        }

        return joinPoint.proceed();
    }

    private String buildRateLimitKey(RateLimit rateLimit) {
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

        return key.toString();
    }

    private String getClientIp() {
        for (String header : IP_HEADERS) {
            String ip = request.getHeader(header);
            if (CharSequenceUtil.isNotBlank(ip) && !"unknown".equalsIgnoreCase(ip)) {
                // X-Forwarded-For 可能包含多个IP，取第一个（客户端真实IP）
                int commaIndex = ip.indexOf(',');
                return commaIndex > 0 ? ip.substring(0, commaIndex).trim() : ip.trim();
            }
        }
        return request.getRemoteAddr();
    }
}
