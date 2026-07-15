package com.pei.dehaze.plugin.dupsubmit.aspect;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.jwt.JWTUtil;
import cn.hutool.jwt.RegisteredPayload;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.plugin.dupsubmit.annotation.PreventDuplicateSubmit;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.util.concurrent.TimeUnit;

/**
 * 处理重复提交的切面
 *
 * @author earthyzinc
 * @since 2.3.0
 */
@Aspect
@Component
@Slf4j
@RequiredArgsConstructor
public class DuplicateSubmitAspect {

    private final RedissonClient redissonClient;
    private static final String RESUBMIT_LOCK_PREFIX = "LOCK:RESUBMIT:";

    /**
     * 防重复提交切点
     */
    @Pointcut("@annotation(preventDuplicateSubmit)")
    public void preventDuplicateSubmitPointCut(PreventDuplicateSubmit preventDuplicateSubmit) {
    }

    @Around(value = "preventDuplicateSubmitPointCut(preventDuplicateSubmit)", argNames = "pjp,preventDuplicateSubmit")
    public Object doAround(ProceedingJoinPoint pjp, PreventDuplicateSubmit preventDuplicateSubmit) throws Throwable {

        String resubmitLockKey = generateResubmitLockKey();
        if (resubmitLockKey != null) {
            long leaseTimeMs = TimeUnit.SECONDS.toMillis(preventDuplicateSubmit.expire());
            RLock lock = redissonClient.getLock(resubmitLockKey);
            boolean lockResult = lock.tryLock(300, leaseTimeMs, TimeUnit.MILLISECONDS);

            if (!lockResult) {
                throw new BusinessException(ResultCode.REPEAT_SUBMIT_ERROR);
            }
            try {
                return pjp.proceed();
            } finally {
                if (lock.isHeldByCurrentThread()) {
                    try {
                        lock.unlock();
                    } catch (IllegalMonitorStateException e) {
                        // 锁已过期，无需解锁
                    }
                }
            }
        }
        return pjp.proceed();
    }

    /**
     * 获取重复提交锁的 key
     */
    private String generateResubmitLockKey() {
        ServletRequestAttributes requestAttributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        String resubmitLockKey = null;
        if (requestAttributes == null) {
            return null;
        }
        HttpServletRequest request = (requestAttributes).getRequest();

        String token = request.getHeader(HttpHeaders.AUTHORIZATION);
        if (CharSequenceUtil.isNotBlank(token) && token.startsWith(SecurityConstants.JWT_TOKEN_PREFIX)) {
            token = token.substring(SecurityConstants.JWT_TOKEN_PREFIX.length());
            // 从 JWT Token 中获取 jti
            String jti = (String) JWTUtil.parseToken(token).getPayload(RegisteredPayload.JWT_ID);
            resubmitLockKey = RESUBMIT_LOCK_PREFIX + jti + ":" + request.getMethod() + "-" + request.getRequestURI();
        }
        return resubmitLockKey;
    }

}
