package com.pei.dehaze.config;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpHeaders;
import org.springframework.mock.web.MockHttpServletRequest;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * AI 转发 multipart 旁路单元测试：白名单内保持原始字节流，白名单外维持默认解析。
 */
@DisplayName("AiProxyMultipartResolver 单元测试")
class AiProxyMultipartResolverTest {

    private final AiProxyMultipartResolver resolver = new AiProxyMultipartResolver();

    @Test
    @DisplayName("skill 上传不经 Spring 解析，原始字节流直达转发层")
    void skillUpload_keepsRawStream() {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/ai/skills/upload");
        request.addHeader(HttpHeaders.CONTENT_TYPE, "multipart/form-data; boundary=boundary-1");

        assertThat(resolver.isMultipart(request)).isFalse();
    }

    @Test
    @DisplayName("其他模块的 multipart 请求维持默认解析")
    void otherMultipart_stillResolved() {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/files/upload");
        request.addHeader(HttpHeaders.CONTENT_TYPE, "multipart/form-data; boundary=boundary-1");

        assertThat(resolver.isMultipart(request)).isTrue();
    }

    @Test
    @DisplayName("非 multipart 请求不受影响")
    void nonMultipart_notResolved() {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/ai/skills/upload");
        request.addHeader(HttpHeaders.CONTENT_TYPE, "application/json");

        assertThat(resolver.isMultipart(request)).isFalse();
    }
}
