package com.pei.dehaze.filter;

import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.filter.OncePerRequestFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;

/**
 * 请求级访问日志过滤器
 * <p>
 * 每请求一条 INFO ACCESS 日志，status/duration_ms/query 为独立字段，
 * traceId/method/path 由 logback MDC 自动注入。
 *
 * @author earthyzinc
 * @since 2023/03/03
 */
@Configuration
@Slf4j
public class RequestLogFilter extends OncePerRequestFilter {

    private static final String ATTR_START_TIME = "reqStartTime";

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        long startTime = System.currentTimeMillis();
        request.setAttribute(ATTR_START_TIME, startTime);

        try {
            filterChain.doFilter(request, response);
        } finally {
            long costMs = System.currentTimeMillis() - startTime;
            String query = request.getQueryString();
            MDC.put("status", String.valueOf(response.getStatus()));
            MDC.put("duration_ms", String.valueOf(costMs));
            if (query != null && !query.isEmpty()) {
                MDC.put("query", query);
            }
            log.info("ACCESS");
            MDC.remove("status");
            MDC.remove("duration_ms");
            MDC.remove("query");
        }
    }
}
