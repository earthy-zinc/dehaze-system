package com.pei.dehaze.plugin.dupsubmit.aspect;

import cn.hutool.core.text.CharSequenceUtil;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.plugin.dupsubmit.annotation.PreventDuplicateSubmit;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.util.concurrent.TimeUnit;

@Aspect
@Component
@Slf4j
@RequiredArgsConstructor
public class DuplicateSubmitAspect {

    private final RedissonClient redissonClient;
    private static final String RESUBMIT_LOCK_PREFIX = "LOCK:RESUBMIT:";

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
                    }
                }
            }
        }
        return pjp.proceed();
    }

    private String generateResubmitLockKey() {
        ServletRequestAttributes requestAttributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (requestAttributes == null) {
            return null;
        }
        HttpServletRequest request = requestAttributes.getRequest();

        String sessionId = null;
        Cookie[] cookies = request.getCookies();
        if (cookies != null) {
            for (Cookie cookie : cookies) {
                if (SecurityConstants.SESSION_COOKIE_NAME.equals(cookie.getName())) {
                    sessionId = cookie.getValue();
                    break;
                }
            }
        }
        if (sessionId == null) {
            sessionId = request.getHeader(SecurityConstants.SESSION_COOKIE_NAME);
        }
        if (CharSequenceUtil.isBlank(sessionId)) {
            return null;
        }
        return RESUBMIT_LOCK_PREFIX + sessionId + ":" + request.getMethod() + "-" + request.getRequestURI();
    }

}
