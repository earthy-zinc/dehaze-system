package com.pei.dehaze.plugin.dupsubmit.filter;

import com.pei.dehaze.plugin.dupsubmit.support.RepeatableBodyRequestWrapper;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * 把可缓存的请求体包装注入过滤链，供防重切面读取请求体摘要。
 *
 * <p>只包装小体积、非表单的请求：multipart 与超大请求体整体缓存会把文件读进内存，
 * 这类请求不包装 ⇒ 防重键按空 body 摘要参与（与 python/go 口径一致）。
 *
 * @author earthyzinc
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 10)
public class RepeatableBodyFilter extends OncePerRequestFilter {

    /**
     * 单请求体缓存上限（字节）：超过则不缓存，避免大请求体常驻内存。
     */
    private static final long MAX_CACHED_BODY_SIZE = 256 * 1024L;

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        if (shouldCache(request)) {
            filterChain.doFilter(new RepeatableBodyRequestWrapper(request), response);
            return;
        }
        filterChain.doFilter(request, response);
    }

    private boolean shouldCache(HttpServletRequest request) {
        String contentType = request.getContentType();
        if (contentType != null && contentType.toLowerCase().startsWith("multipart/")) {
            return false;
        }
        long contentLength = request.getContentLengthLong();
        // 无 Content-Length（分块传输）时无法预判体积，按小请求处理
        return contentLength <= 0 || contentLength <= MAX_CACHED_BODY_SIZE;
    }
}
