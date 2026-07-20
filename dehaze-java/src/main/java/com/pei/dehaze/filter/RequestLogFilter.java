package com.pei.dehaze.filter;

import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.filter.CommonsRequestLoggingFilter;

/**
 * 请求日志打印过滤器
 * <p>
 * 记录请求 URI、查询参数、客户端信息、请求体（截断）和响应耗时，
 * 便于生产环境问题排查。
 *
 * @author earthyzinc
 * @since 2023/03/03
 */
@Configuration
@Slf4j
public class RequestLogFilter extends CommonsRequestLoggingFilter {

    private static final String ATTR_START_TIME = "reqStartTime";

    public RequestLogFilter() {
        setIncludeClientInfo(true);
        setIncludeQueryString(true);
        setIncludePayload(true);
        setMaxPayloadLength(1024);
        setIncludeHeaders(false);
    }

    @Override
    protected boolean shouldLog(HttpServletRequest request) {
        // 以 info 级别输出请求日志
        return this.logger.isInfoEnabled();
    }

    @Override
    protected void beforeRequest(HttpServletRequest request, String message) {
        request.setAttribute(ATTR_START_TIME, System.currentTimeMillis());
        String requestURI = request.getRequestURI();
        String queryString = request.getQueryString();
        String clientInfo = request.getRemoteAddr();
        log.info("请求开始: {} {}{} client={}",
                request.getMethod(),
                requestURI,
                queryString != null ? "?" + queryString : "",
                clientInfo);
        super.beforeRequest(request, message);
    }

    @Override
    protected void afterRequest(HttpServletRequest request, String message) {
        Long startTime = (Long) request.getAttribute(ATTR_START_TIME);
        long costMs = startTime != null ? System.currentTimeMillis() - startTime : -1;
        log.info("请求结束: {} {} 耗时={}ms", request.getMethod(), request.getRequestURI(), costMs);
        super.afterRequest(request, message);
    }
}
