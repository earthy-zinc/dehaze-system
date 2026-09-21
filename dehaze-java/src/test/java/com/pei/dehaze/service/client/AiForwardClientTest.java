package com.pei.dehaze.service.client;

import com.pei.dehaze.config.property.AlgorithmProperties;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.ArgumentMatchers;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpHeaders;

import java.io.IOException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Flow;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * AI 转发客户端单元测试：请求头组装、缓冲区透传与 SSE 流式管道（取消/超时/上游异常）。
 */
@DisplayName("AiForwardClient 转发客户端单元测试")
@ExtendWith(MockitoExtension.class)
class AiForwardClientTest {

    private static final String BASE_URL = "http://127.0.0.1:8991";

    @Mock
    private HttpClient httpClient;

    private AlgorithmProperties props;
    private AiForwardClient client;

    @BeforeEach
    void setUp() {
        props = new AlgorithmProperties();
        props.setBaseUrl(BASE_URL);
        props.setReadTimeout(120000);
        props.setAiProxyStreamIdleTimeout(300000);
        client = new AiForwardClient(props, httpClient);
    }

    @Test
    @DisplayName("exchange - 原样透传上游状态码、Content-Type 与信封响应体")
    void exchange_passesThroughEnvelope() throws Exception {
        byte[] envelope = "{\"code\":\"00000\",\"msg\":\"一切ok\",\"data\":{\"id\":1}}"
                .getBytes(StandardCharsets.UTF_8);
        HttpResponse<byte[]> upstream = byteResponse(200, "application/json", envelope);
        when(httpClient.send(any(HttpRequest.class), ArgumentMatchers.<HttpResponse.BodyHandler<byte[]>>any()))
                .thenReturn(upstream);

        AiForwardClient.BufferedResponse response = client.exchange(
                "POST", "/api/v1/ai/messages/34/stop", sessionHeaders(), HttpRequest.BodyPublishers.noBody());

        assertThat(response.status()).isEqualTo(200);
        assertThat(response.contentType()).isEqualTo("application/json");
        assertThat(response.body()).isEqualTo(envelope);
    }

    @Test
    @DisplayName("exchange - 请求组装：baseUrl + path、方法、会话头与读取超时")
    void exchange_buildsUpstreamRequest() throws Exception {
        HttpResponse<byte[]> upstream = byteResponse(200, "application/json", "{}".getBytes(StandardCharsets.UTF_8));
        when(httpClient.send(any(HttpRequest.class), ArgumentMatchers.<HttpResponse.BodyHandler<byte[]>>any()))
                .thenReturn(upstream);

        client.exchange("POST", "/api/v1/ai/agents/5/eval/runs?mode=full", sessionHeaders(),
                HttpRequest.BodyPublishers.ofString("{\"datasetId\":1}"));

        HttpRequest sent = captureRequest();
        assertThat(sent.uri().toString()).isEqualTo(BASE_URL + "/api/v1/ai/agents/5/eval/runs?mode=full");
        assertThat(sent.method()).isEqualTo("POST");
        assertThat(sent.headers().firstValue("Authorization")).contains("Bearer session-token");
        assertThat(sent.headers().firstValue("X-Session-Id")).contains("sess-1");
        assertThat(sent.timeout()).contains(java.time.Duration.ofMillis(120000));
        assertThat(sent.bodyPublisher()).isPresent();
    }

    @Test
    @DisplayName("stream - SSE 分块逐段交付，保留状态码与 Content-Type")
    void stream_forwardsChunksInOrder() throws Exception {
        FakeBodyPublisher publisher = new FakeBodyPublisher();
        stubPublisherResponse(200, "text/event-stream", publisher);

        List<String> received = new ArrayList<>();
        int[] status = new int[1];
        String[] contentType = new String[1];
        client.stream("POST", "/api/v1/ai/conversations/1/messages", sessionHeaders(),
                HttpRequest.BodyPublishers.noBody(), (code, type, body) -> {
                    status[0] = code;
                    contentType[0] = type;
                    publisher.emit("event: message.start\ndata: {}\n\n");
                    publisher.emit(": ping\n\n");
                    publisher.emit("data: [DONE]\n\n");
                    publisher.complete();
                    received.add(new String(body.readAllBytes(), StandardCharsets.UTF_8));
                });

        assertThat(status[0]).isEqualTo(200);
        assertThat(contentType[0]).isEqualTo("text/event-stream");
        // 心跳注释行与事件块原样透传，顺序不变
        assertThat(received).containsExactly(
                "event: message.start\ndata: {}\n\n" + ": ping\n\n" + "data: [DONE]\n\n");
        assertThat(publisher.cancelled).isTrue();
    }

    @Test
    @DisplayName("stream - 客户端断连（消费端 IOException）时取消上游订阅")
    void stream_cancelsUpstreamWhenClientDisconnects() throws Exception {
        FakeBodyPublisher publisher = new FakeBodyPublisher();
        stubPublisherResponse(200, "text/event-stream", publisher);

        assertThatThrownBy(() -> client.stream("POST", "/api/v1/ai/messages/9/regenerate", sessionHeaders(),
                HttpRequest.BodyPublishers.noBody(), (code, type, body) -> {
                    throw new IOException("客户端已断开");
                }))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("客户端已断开");

        assertThat(publisher.cancelled).isTrue();
    }

    @Test
    @DisplayName("stream - 上游长时间无数据抛空闲超时，避免转发线程被卡死")
    void stream_idleTimeoutFailsStream() throws Exception {
        props.setAiProxyStreamIdleTimeout(80);
        FakeBodyPublisher publisher = new FakeBodyPublisher();
        stubPublisherResponse(200, "text/event-stream", publisher);

        assertThatThrownBy(() -> client.stream("GET", "/api/v1/ai/conversations/1/messages/stream/s-1",
                sessionHeaders(), HttpRequest.BodyPublishers.noBody(),
                (code, type, body) -> body.read()))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("空闲超时");

        assertThat(publisher.cancelled).isTrue();
    }

    @Test
    @DisplayName("stream - 上游流异常向消费端传播")
    void stream_propagatesUpstreamError() throws Exception {
        FakeBodyPublisher publisher = new FakeBodyPublisher();
        stubPublisherResponse(200, "text/event-stream", publisher);

        assertThatThrownBy(() -> client.stream("POST", "/api/v1/ai/conversations/1/messages", sessionHeaders(),
                HttpRequest.BodyPublishers.noBody(), (code, type, body) -> {
                    publisher.fail(new IllegalStateException("上游连接中断"));
                    body.read();
                }))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("上游连接中断");
    }

    private HttpHeaders sessionHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.set(HttpHeaders.AUTHORIZATION, "Bearer session-token");
        headers.set("X-Session-Id", "sess-1");
        return headers;
    }

    private HttpRequest captureRequest() throws Exception {
        ArgumentCaptor<HttpRequest> captor = ArgumentCaptor.forClass(HttpRequest.class);
        verify(httpClient).send(captor.capture(), ArgumentMatchers.<HttpResponse.BodyHandler<byte[]>>any());
        return captor.getValue();
    }

    @SuppressWarnings("unchecked")
    private static HttpResponse<byte[]> byteResponse(int status, String contentType, byte[] body) {
        HttpResponse<byte[]> response = org.mockito.Mockito.mock(HttpResponse.class);
        when(response.statusCode()).thenReturn(status);
        when(response.headers()).thenReturn(upstreamHeaders(contentType));
        when(response.body()).thenReturn(body);
        return response;
    }

    @SuppressWarnings("unchecked")
    private static HttpResponse<Flow.Publisher<List<ByteBuffer>>> publisherResponse(
            int status, String contentType, FakeBodyPublisher publisher) {
        HttpResponse<Flow.Publisher<List<ByteBuffer>>> response = org.mockito.Mockito.mock(HttpResponse.class);
        when(response.statusCode()).thenReturn(status);
        when(response.headers()).thenReturn(upstreamHeaders(contentType));
        when(response.body()).thenReturn(publisher);
        return response;
    }

    private void stubPublisherResponse(int status, String contentType, FakeBodyPublisher publisher) throws Exception {
        HttpResponse<Flow.Publisher<List<ByteBuffer>>> response = publisherResponse(status, contentType, publisher);
        when(httpClient.send(any(HttpRequest.class),
                ArgumentMatchers.<HttpResponse.BodyHandler<Flow.Publisher<List<ByteBuffer>>>>any()))
                .thenReturn(response);
    }

    private static java.net.http.HttpHeaders upstreamHeaders(String contentType) {
        return java.net.http.HttpHeaders.of(Map.of("content-type", List.of(contentType)), (name, value) -> true);
    }

    /**
     * 上游响应体 publisher 的测试替身：记录取消动作，允许测试主动推送数据块。
     */
    private static final class FakeBodyPublisher implements Flow.Publisher<List<ByteBuffer>> {

        private Flow.Subscriber<? super List<ByteBuffer>> subscriber;
        private boolean cancelled;

        @Override
        public void subscribe(Flow.Subscriber<? super List<ByteBuffer>> subscriber) {
            this.subscriber = subscriber;
            subscriber.onSubscribe(new Flow.Subscription() {
                @Override
                public void request(long n) {
                }

                @Override
                public void cancel() {
                    cancelled = true;
                }
            });
        }

        void emit(String chunk) {
            subscriber.onNext(List.of(ByteBuffer.wrap(chunk.getBytes(StandardCharsets.UTF_8))));
        }

        void complete() {
            subscriber.onComplete();
        }

        void fail(Throwable error) {
            subscriber.onError(error);
        }
    }
}
