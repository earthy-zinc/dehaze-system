package com.pei.dehaze.aspect;

import cn.hutool.core.text.CharSequenceUtil;
import com.pei.dehaze.annotation.AuditLog;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AuditLogService;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.core.DefaultParameterNameDiscoverer;
import org.springframework.expression.spel.standard.SpelExpressionParser;
import org.springframework.expression.spel.support.StandardEvaluationContext;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.lang.reflect.Method;

@Aspect
@Component
@Slf4j
@RequiredArgsConstructor
public class AuditLogAspect {

    private final AuditLogService auditLogService;

    private final SpelExpressionParser spelExpressionParser = new SpelExpressionParser();
    private final DefaultParameterNameDiscoverer parameterNameDiscoverer = new DefaultParameterNameDiscoverer();

    @Around("@annotation(auditLog)")
    public Object around(ProceedingJoinPoint joinPoint, AuditLog auditLog) throws Throwable {
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();
        Object[] args = joinPoint.getArgs();
        String[] parameterNames = parameterNameDiscoverer.getParameterNames(method);

        Object beforeValue = evaluateSpel(auditLog.beforeSpel(), args, parameterNames, null, method, "beforeSpel");

        Object result = null;
        try {
            result = joinPoint.proceed();
            return result;
        } finally {
            try {
                Long operatorId = SecurityUtils.getUserId();
                HttpServletRequest request = getRequest();
                String ip = request != null ? getClientIp(request) : null;
                String userAgent = request != null ? request.getHeader("User-Agent") : null;

                Object targetId = evaluateSpel(auditLog.targetIdSpel(), args, parameterNames, result, method, "targetIdSpel");
                Object afterValue = evaluateSpel(auditLog.afterSpel(), args, parameterNames, result, method, "afterSpel");

                auditLogService.recordAudit(operatorId, auditLog.targetType(), targetId,
                        auditLog.action(), auditLog.module(), beforeValue, afterValue, ip, userAgent);
            } catch (Exception e) {
                log.warn("记录审计日志失败: method={}", method.getName(), e);
            }
        }
    }

    private Object evaluateSpel(String spel, Object[] args, String[] parameterNames, Object result, Method method, String field) {
        if (CharSequenceUtil.isBlank(spel)) {
            return null;
        }
        try {
            StandardEvaluationContext context = new StandardEvaluationContext(result);
            if (parameterNames != null) {
                for (int i = 0; i < parameterNames.length && i < args.length; i++) {
                    context.setVariable(parameterNames[i], args[i]);
                }
            }
            context.setVariable("result", result);
            return spelExpressionParser.parseExpression(spel).getValue(context);
        } catch (Exception e) {
            log.warn("解析 SpEL 失败: field={}, expr={}, method={}", field, spel, method.getName(), e);
            return null;
        }
    }

    private HttpServletRequest getRequest() {
        ServletRequestAttributes requestAttributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        return requestAttributes != null ? requestAttributes.getRequest() : null;
    }

    private String getClientIp(HttpServletRequest request) {
        String ip = request.getHeader("X-Forwarded-For");
        if (CharSequenceUtil.isBlank(ip) || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("X-Real-IP");
        }
        if (CharSequenceUtil.isBlank(ip) || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getRemoteAddr();
        }
        if (ip != null && ip.contains(",")) {
            ip = ip.split(",")[0].trim();
        }
        return ip;
    }
}
