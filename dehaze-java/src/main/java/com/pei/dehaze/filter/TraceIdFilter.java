package com.pei.dehaze.filter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.MDC;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.UUID;

/**
 * 请求追踪 ID 过滤器
 * <p>
 * 在请求入口生成或读取上游传入的 X-Trace-Id，写入 SLF4J MDC，
 * 使整条请求链路的日志都能输出 traceId/method/path，并在响应头回传。
 *
 * @author earthyzinc
 * @since 2026-07-13
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class TraceIdFilter extends OncePerRequestFilter {

    public static final String TRACE_ID_HEADER = "X-Trace-Id";
    public static final String MDC_TRACE_ID = "trace_id";
    public static final String MDC_METHOD = "method";
    public static final String MDC_PATH = "path";
    public static final String MDC_IP = "ip";
    public static final String MDC_USER_AGENT = "user_agent";

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        // 优先读取上游透传的 traceId，缺失时生成新的
        String traceId = request.getHeader(TRACE_ID_HEADER);
        if (traceId == null || traceId.isBlank()) {
            traceId = UUID.randomUUID().toString().replace("-", "");
        }

        // 采集客户端 IP（优先 X-Forwarded-For 首段，适配 Nginx 反代），无则取直连 IP
        String forwarded = request.getHeader("X-Forwarded-For");
        String ip = "";
        if (forwarded != null && !forwarded.isEmpty()) {
            ip = forwarded.split(",")[0].trim();
        } else if (request.getRemoteAddr() != null) {
            ip = request.getRemoteAddr();
        }
        String userAgent = request.getHeader("User-Agent");

        MDC.put(MDC_TRACE_ID, traceId);
        MDC.put(MDC_METHOD, request.getMethod());
        MDC.put(MDC_PATH, request.getRequestURI());
        MDC.put(MDC_IP, ip);
        MDC.put(MDC_USER_AGENT, userAgent);

        try {
            // 响应头回传 traceId，便于调用方关联
            response.setHeader(TRACE_ID_HEADER, traceId);
            filterChain.doFilter(request, response);
        } finally {
            // 必须清理 MDC，防止线程池复用导致字段串号
            MDC.remove(MDC_TRACE_ID);
            MDC.remove(MDC_METHOD);
            MDC.remove(MDC_PATH);
            MDC.remove(MDC_IP);
            MDC.remove(MDC_USER_AGENT);
        }
    }
}
