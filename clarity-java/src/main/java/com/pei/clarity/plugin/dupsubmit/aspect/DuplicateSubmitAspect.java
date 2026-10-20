package com.pei.clarity.plugin.dupsubmit.aspect;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.crypto.digest.DigestUtil;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.plugin.dupsubmit.annotation.PreventDuplicateSubmit;
import com.pei.clarity.plugin.dupsubmit.support.RepeatableBodyRequestWrapper;
import com.pei.clarity.security.util.SecurityUtils;
import jakarta.servlet.http.HttpServletRequest;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.redisson.api.RBucket;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

/**
 * 防重复提交切面。
 *
 * <p>键格式（三端共享同一 Redis，**格式即跨端契约**，见
 * {@code clarity-doc/docs/05-改造计划/防重复提交三端统一改造计划.md}）：
 *
 * <pre>anti_repeat:{identity}:{METHOD}:{path}:{sortedQuery}:{bodyHash}</pre>
 *
 * <p>参与防重的端点由 {@link PreventDuplicateSubmit} 显式声明（白名单），
 * 存活集合须与 python 中间件白名单、go 路由挂载完全一致（由跨端契约测试对拍）。
 *
 * @author earthyzinc
 */
@Aspect
@Component
@Slf4j
@RequiredArgsConstructor
public class DuplicateSubmitAspect {

    private static final String RESUBMIT_KEY_PREFIX = "anti_repeat:";

    private final RedissonClient redissonClient;

    @Pointcut("@annotation(preventDuplicateSubmit)")
    public void preventDuplicateSubmitPointCut(PreventDuplicateSubmit preventDuplicateSubmit) {}

    @Around(value = "preventDuplicateSubmitPointCut(preventDuplicateSubmit)", argNames = "pjp,preventDuplicateSubmit")
    public Object doAround(ProceedingJoinPoint pjp, PreventDuplicateSubmit preventDuplicateSubmit) throws Throwable {

        String resubmitLockKey = buildResubmitKey();
        if (resubmitLockKey == null) {
            return pjp.proceed();
        }

        // SET NX EX 语义（与 python/go 同构）：键在过期窗口内只允许写入一次，业务完成后**不释放**。
        // 旧实现用 Redisson 锁 + finally unlock，等价于"仅并发去重"——同一用户串行两次点击不再被拦。
        RBucket<String> bucket = redissonClient.getBucket(resubmitLockKey);
        boolean acquired = bucket.trySet("1", preventDuplicateSubmit.expire(), TimeUnit.SECONDS);
        if (!acquired) {
            log.warn("防重复提交拦截: key={}", resubmitLockKey);
            throw new BusinessException(ResultCode.REPEAT_SUBMIT_ERROR);
        }
        return pjp.proceed();
    }

    /**
     * 键的五段固定（query / bodyHash 为空时保留空段），保证三端对同一请求算出同一键。
     */
    private String buildResubmitKey() {
        ServletRequestAttributes requestAttributes =
                (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (requestAttributes == null) {
            return null;
        }
        HttpServletRequest request = requestAttributes.getRequest();
        return RESUBMIT_KEY_PREFIX
                + resolveIdentity(request)
                + ":" + request.getMethod()
                + ":" + request.getRequestURI()
                + ":" + normalizeQuery(request.getQueryString())
                + ":" + resolveBodyHash(request);
    }

    /**
     * 身份统一收敛到用户维度：sessionId / JWT jti 都随登录态变化，
     * 会让"同一用户重新登录后提交同一请求"各自算出新键而绕过防重。
     */
    private String resolveIdentity(HttpServletRequest request) {
        Long userId = SecurityUtils.getUserId();
        if (userId != null) {
            return "user:" + userId;
        }
        return "ip:" + resolveClientIp(request);
    }

    private String resolveClientIp(HttpServletRequest request) {
        String forwarded = request.getHeader("X-Forwarded-For");
        if (CharSequenceUtil.isNotBlank(forwarded)) {
            return forwarded.split(",")[0].trim();
        }
        return request.getRemoteAddr();
    }

    /**
     * multipart 与未缓存（超大）请求体按空摘要参与，与 python/go 口径一致。
     */
    private String resolveBodyHash(HttpServletRequest request) {
        if (request instanceof RepeatableBodyRequestWrapper wrapper) {
            byte[] body = wrapper.getCachedBody();
            if (body.length > 0) {
                return DigestUtil.md5Hex(body);
            }
        }
        return "";
    }

    /**
     * query 按 (key, value) 全排序后重新编码，避免 {@code ?a=1&b=2} 与 {@code ?b=2&a=1} 绕过防重。
     */
    private String normalizeQuery(String rawQuery) {
        if (CharSequenceUtil.isBlank(rawQuery)) {
            return "";
        }
        List<String[]> pairs = new ArrayList<>();
        for (String pair : rawQuery.split("&")) {
            if (pair.isEmpty()) {
                continue;
            }
            int index = pair.indexOf('=');
            String rawKey = index >= 0 ? pair.substring(0, index) : pair;
            String rawValue = index >= 0 ? pair.substring(index + 1) : "";
            pairs.add(new String[] {decode(rawKey), decode(rawValue)});
        }
        pairs.sort(Comparator.comparing((String[] pair) -> pair[0]).thenComparing(pair -> pair[1]));
        return pairs.stream()
                .map(pair -> encode(pair[0]) + "=" + encode(pair[1]))
                .collect(Collectors.joining("&"));
    }

    private String decode(String value) {
        return URLDecoder.decode(value, StandardCharsets.UTF_8);
    }

    private String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
    }
}
