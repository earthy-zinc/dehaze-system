package com.pei.dehaze.controller;

import cn.hutool.json.JSONUtil;
import com.pei.dehaze.filter.TraceIdFilter;
import com.pei.dehaze.service.client.AiForwardClient;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.MDC;
import org.springframework.http.HttpHeaders;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.ConnectException;
import java.net.http.HttpRequest;
import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * AI 转发 Controller 单元测试：白名单 404、会话头透传、信封原样透传、SSE 管道与不可达降级。
 */
@DisplayName("AiProxyController 转发入口单元测试")
@ExtendWith(MockitoExtension.class)
class AiProxyControllerTest {

    @Mock
    private AiForwardClient aiForwardClient;

    private AiProxyController controller;

    @BeforeEach
    void setUp() {
        controller = new AiProxyController(aiForwardClient);
    }

    @Test
    @DisplayName("forwardHeaders - 透传会话身份与协议头，不携带逐跳头")
    void forwardHeaders_passesSessionAndProtocolHeaders() {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/ai/conversations/1/messages");
        request.addHeader(HttpHeaders.AUTHORIZATION, "Bearer session-token");
        request.addHeader(HttpHeaders.COOKIE, "X-Session-Id=cookie-sess");
        request.addHeader("X-Session-Id", "header-sess");
        // Claude 协议凭据载体，python 侧 API Key 中间件据此鉴权
        request.addHeader("x-api-key", "sk-ant-sess");
        request.addHeader(HttpHeaders.CONTENT_TYPE, "application/json");
        request.addHeader(HttpHeaders.ACCEPT, "text/event-stream");
        request.addHeader("Idempotency-Key", "idem-1");
        request.addHeader("Last-Event-ID", "42");
        request.addHeader("Content-Length", "128");
        request.addHeader("Connection", "keep-alive");
        request.addHeader("Host", "127.0.0.1:8989");
        // 客户端自报的转发链不被采信
        request.addHeader("X-Forwarded-For", "1.2.3.4");

        HttpHeaders headers;
        try {
            MDC.put(TraceIdFilter.MDC_IP, "203.0.113.7");
            MDC.put(TraceIdFilter.MDC_USER_AGENT, "claude-sdk/1.0");
            headers = AiProxyController.forwardHeaders(request);
        } finally {
            MDC.clear();
        }
        // python 侧限流/审计按客户端 IP 与 UA 计，需透传 java 解析出的单值
        assertThat(headers.getFirst("X-Forwarded-For")).isEqualTo("203.0.113.7");
        assertThat(headers.getFirst(HttpHeaders.USER_AGENT)).isEqualTo("claude-sdk/1.0");

        assertThat(headers.getFirst(HttpHeaders.AUTHORIZATION)).isEqualTo("Bearer session-token");
        assertThat(headers.getFirst(HttpHeaders.COOKIE)).isEqualTo("X-Session-Id=cookie-sess");
        assertThat(headers.getFirst("X-Session-Id")).isEqualTo("header-sess");
        assertThat(headers.getFirst("x-api-key")).isEqualTo("sk-ant-sess");
        assertThat(headers.getFirst(HttpHeaders.CONTENT_TYPE)).isEqualTo("application/json");
        assertThat(headers.getFirst(HttpHeaders.ACCEPT)).isEqualTo("text/event-stream");
        assertThat(headers.getFirst("Idempotency-Key")).isEqualTo("idem-1");
        assertThat(headers.getFirst("Last-Event-ID")).isEqualTo("42");
        assertThat(headers.containsKey("Content-Length")).isFalse();
        assertThat(headers.containsKey("Connection")).isFalse();
        assertThat(headers.containsKey("Host")).isFalse();
    }

    @Test
    @DisplayName("白名单外路径返回 404 且不触达上游")
    void forward_rejectsPathOutsideWhitelist() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/api/v1/ai/unknown/probe");
        MockHttpServletResponse response = new MockHttpServletResponse();

        controller.forward(request, response);

        assertThat(response.getStatus()).isEqualTo(404);
        assertThat(JSONUtil.parseObj(response.getContentAsString()).getStr("code")).isEqualTo("A0401");
        verifyNoInteractions(aiForwardClient);
    }

    @Test
    @DisplayName("JSON 端点原样透传上游状态码、Content-Type 与信封响应体")
    void forward_passesThroughEnvelopeBody() throws Exception {
        byte[] envelope = "{\"code\":\"00000\",\"msg\":\"一切ok\",\"data\":{\"stopped\":true}}"
                .getBytes(StandardCharsets.UTF_8);
        when(aiForwardClient.exchange(anyString(), anyString(), any(HttpHeaders.class),
                any(HttpRequest.BodyPublisher.class)))
                .thenReturn(new AiForwardClient.BufferedResponse(200, "application/json", envelope));
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/ai/messages/34/stop");
        MockHttpServletResponse response = new MockHttpServletResponse();

        controller.forward(request, response);

        assertThat(response.getStatus()).isEqualTo(200);
        assertThat(response.getContentType()).isEqualTo("application/json");
        assertThat(response.getContentAsByteArray()).isEqualTo(envelope);
    }

    @Test
    @DisplayName("POST 请求体整体转发，并透传目标路径")
    void forward_forwardsJsonBody() throws Exception {
        when(aiForwardClient.exchange(anyString(), anyString(), any(HttpHeaders.class),
                any(HttpRequest.BodyPublisher.class)))
                .thenReturn(new AiForwardClient.BufferedResponse(200, "application/json", "{}".getBytes()));
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/ai/agents/5/test");
        request.setContent("{\"prompt\":\"hi\"}".getBytes(StandardCharsets.UTF_8));
        MockHttpServletResponse response = new MockHttpServletResponse();

        controller.forward(request, response);

        ArgumentCaptor<HttpRequest.BodyPublisher> captor = ArgumentCaptor.forClass(HttpRequest.BodyPublisher.class);
        verify(aiForwardClient).exchange(eq("POST"), eq("/api/v1/ai/agents/5/test"), any(HttpHeaders.class),
                captor.capture());
        assertThat(captor.getValue().contentLength()).isEqualTo("{\"prompt\":\"hi\"}".length());
    }

    @Test
    @DisplayName("multipart 端点以原始字节流转发，保留 Content-Type 与 boundary")
    void forward_streamsRawMultipartBody() throws Exception {
        byte[] raw = ("--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"demo.zip\"\r\n"
                + "\r\nzip-bytes\r\n--boundary--\r\n").getBytes(StandardCharsets.UTF_8);
        when(aiForwardClient.exchange(anyString(), anyString(), any(HttpHeaders.class),
                any(HttpRequest.BodyPublisher.class)))
                .thenReturn(new AiForwardClient.BufferedResponse(200, "application/json", "{}".getBytes()));
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/ai/skills/upload");
        request.addHeader(HttpHeaders.CONTENT_TYPE, "multipart/form-data; boundary=boundary");
        request.setContent(raw);
        MockHttpServletResponse response = new MockHttpServletResponse();

        controller.forward(request, response);

        ArgumentCaptor<HttpHeaders> headersCaptor = ArgumentCaptor.forClass(HttpHeaders.class);
        ArgumentCaptor<HttpRequest.BodyPublisher> bodyCaptor = ArgumentCaptor.forClass(HttpRequest.BodyPublisher.class);
        verify(aiForwardClient).exchange(eq("POST"), eq("/api/v1/ai/skills/upload"), headersCaptor.capture(),
                bodyCaptor.capture());
        assertThat(headersCaptor.getValue().getFirst(HttpHeaders.CONTENT_TYPE))
                .isEqualTo("multipart/form-data; boundary=boundary");
        assertThat(bodyCaptor.getValue().contentLength()).isEqualTo(raw.length);
    }

    @Test
    @DisplayName("SSE 端点走流式管道，事件块原样写回客户端")
    void forward_pipesSseStream() throws Exception {
        byte[] events = "event: content_block.delta\ndata: {\"text\":\"你好\"}\n\n: ping\n\n"
                .getBytes(StandardCharsets.UTF_8);
        doAnswer(invocation -> {
            AiForwardClient.StreamPipe pipe = invocation.getArgument(4);
            pipe.pipe(200, "text/event-stream", new ByteArrayInputStream(events));
            return null;
        }).when(aiForwardClient).stream(anyString(), anyString(), any(HttpHeaders.class),
                any(HttpRequest.BodyPublisher.class), any(AiForwardClient.StreamPipe.class));
        MockHttpServletRequest request = new MockHttpServletRequest(
                "POST", "/api/v1/ai/conversations/1/messages");
        MockHttpServletResponse response = new MockHttpServletResponse();

        controller.forward(request, response);

        assertThat(response.getStatus()).isEqualTo(200);
        assertThat(response.getContentType()).isEqualTo("text/event-stream");
        assertThat(response.getHeader(HttpHeaders.CACHE_CONTROL)).isEqualTo("no-cache");
        assertThat(response.getContentAsByteArray()).isEqualTo(events);
    }

    @Test
    @DisplayName("python 不可达返回统一业务错误信封而非裸 500")
    void forward_returnsBusinessErrorWhenUpstreamUnreachable() throws Exception {
        when(aiForwardClient.exchange(anyString(), anyString(), any(HttpHeaders.class),
                any(HttpRequest.BodyPublisher.class)))
                .thenThrow(new ConnectException("Connection refused"));
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/api/v1/ai/mcp/servers/7/health");
        MockHttpServletResponse response = new MockHttpServletResponse();

        controller.forward(request, response);

        assertThat(response.getStatus()).isEqualTo(503);
        assertThat(JSONUtil.parseObj(response.getContentAsString()).getStr("code")).isEqualTo("C0001");
    }

    @Test
    @DisplayName("SSE 已输出首块后上游断开：只断流，不再改写响应")
    void forward_doesNotRewriteCommittedResponse() throws Exception {
        doThrow(new IOException("上游连接中断")).when(aiForwardClient).stream(anyString(), anyString(),
                any(HttpHeaders.class), any(HttpRequest.BodyPublisher.class),
                any(AiForwardClient.StreamPipe.class));
        MockHttpServletRequest request = new MockHttpServletRequest(
                "POST", "/api/v1/ai/conversations/1/messages");
        MockHttpServletResponse response = new MockHttpServletResponse() {
            @Override
            public boolean isCommitted() {
                return true;
            }
        };

        controller.forward(request, response);

        assertThat(response.getContentAsByteArray()).isEmpty();
        assertThat(response.getStatus()).isEqualTo(200);
    }
}
