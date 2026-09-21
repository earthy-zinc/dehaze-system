package com.pei.dehaze.config;

import com.pei.dehaze.service.client.AiProxyRoutes;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.support.StandardServletMultipartResolver;

/**
 * AI 转发白名单内的 multipart 请求不交由 Spring 解析。
 * <p>
 * 解析会消费原始请求体并把文件落到临时目录，导致无法把文件流原样转发给 dehaze-python；
 * 白名单外的请求维持默认解析行为，其他模块的文件上传不受影响。
 *
 * @author earthyzinc
 * @since 2026-09-17
 */
@Component("multipartResolver")
public class AiProxyMultipartResolver extends StandardServletMultipartResolver {

    @Override
    public boolean isMultipart(HttpServletRequest request) {
        return !AiProxyRoutes.isRawMultipart(request.getMethod(), AiProxyRoutes.pathOf(request))
                && super.isMultipart(request);
    }
}
