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
            int expire = preventDuplicateSubmit.expire(); // 防重提交锁过期时间
            RLock lock = redissonClient.getLock(resubmitLockKey);
            boolean lockResult = false;
            int retryTimes = 3; // 重试次数
            for (int i = 0; i < retryTimes; i++) {
                lockResult = lock.tryLock(0, expire, TimeUnit.SECONDS);
                if (lockResult) {
                    break;
                }
                // 等待一段时间后重试，减少立即重试带来的系统压力
                Thread.sleep(100);
            }

            if (!lockResult) {
                log.error("多次尝试获取锁失败，lock key: {}", resubmitLockKey);
                throw new BusinessException(ResultCode.REPEAT_SUBMIT_ERROR);
            }
            try {
                return pjp.proceed();
            } catch (Throwable t) {
                log.error("方法执行异常, lock key: {}", resubmitLockKey, t);
                throw t;
            } finally {
                lock.unlock();
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
            return resubmitLockKey;
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
