package com.pei.dehaze.service.client;

import com.pei.dehaze.config.property.AlgorithmProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Flow;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * dehaze-python 转发客户端。
 * <p>
 * 与 {@link PythonAlgorithmClient} 不同，本客户端不做重试：转发的是推理 / SSE / 文件上传类请求，
 * 重试会造成重复推理与重复计费；失败由客户端按业务语义自行重试（如 SSE 断线重连）。
 *
 * @author earthyzinc
 * @since 2026-09-17
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class AiForwardClient {

    private final AlgorithmProperties props;
    private final HttpClient aiProxyHttpClient;

    /** 非流式转发结果：状态码、Content-Type 与响应体原样回传 */
    public record BufferedResponse(int status, String contentType, byte[] body) {
    }

    /**
     * 流式转发回调：上游响应就绪后由调用方消费响应体
     */
    @FunctionalInterface
    public interface StreamPipe {
        /**
         * @param status      上游 HTTP 状态码
         * @param contentType 上游 Content-Type（可能为空）
         * @param body        上游响应体流；实现必须读尽或抛出 IOException
         */
        void pipe(int status, String contentType, InputStream body) throws IOException;
    }

    /**
     * 非流式转发：整体读取上游响应，保留状态码与 Content-Type。
     * <p>
     * 上游未响应前（连接/首字节阶段）失败会抛出 IOException，调用方据此返回统一业务错误。
     */
    public BufferedResponse exchange(String method, String path, HttpHeaders headers, HttpRequest.BodyPublisher body)
            throws IOException, InterruptedException {
        HttpRequest request = buildRequest(method, path, headers, body, props.getReadTimeout());
        HttpResponse<byte[]> response = aiProxyHttpClient.send(request, HttpResponse.BodyHandlers.ofByteArray());
        return new BufferedResponse(response.statusCode(), contentTypeOf(response), response.body());
    }

    /**
     * 流式转发（SSE）：上游响应体逐块交付给 pipe；pipe 抛出 IOException（客户端已断连）时取消上游订阅，
     * 避免 python 推理继续空转烧算力。
     * <p>
     * 空闲超时在下游读取循环内生效——HttpRequest 的 timeout 只约束"响应头到达"，
     * 对已建立的流式响应体不再生效（见 JDK HttpClient 语义）。
     */
    public void stream(String method, String path, HttpHeaders headers, HttpRequest.BodyPublisher body, StreamPipe pipe)
            throws IOException, InterruptedException {
        long idleTimeout = props.getAiProxyStreamIdleTimeout();
        HttpRequest request = buildRequest(method, path, headers, body, idleTimeout);
        HttpResponse<Flow.Publisher<List<ByteBuffer>>> response =
                aiProxyHttpClient.send(request, HttpResponse.BodyHandlers.ofPublisher());
        try (UpstreamStream upstream = new UpstreamStream(idleTimeout)) {
            response.body().subscribe(upstream);
            pipe.pipe(response.statusCode(), contentTypeOf(response), upstream);
        }
    }

    private HttpRequest buildRequest(String method, String path, HttpHeaders headers,
                                     HttpRequest.BodyPublisher body, long timeoutMs) {
        HttpRequest.Builder builder;
        try {
            builder = HttpRequest.newBuilder(URI.create(props.getBaseUrl() + path));
        } catch (IllegalArgumentException e) {
            // 请求路径含 URI 非法字符（如未编码的花括号），Python 端同样无法解析
            throw new IllegalArgumentException("非法请求路径: " + path, e);
        }
        builder.method(method, body).timeout(Duration.ofMillis(timeoutMs));
        headers.forEach((name, values) -> values.forEach(value -> builder.header(name, value)));
        return builder.build();
    }

    private static String contentTypeOf(HttpResponse<?> response) {
        return response.headers().firstValue(HttpHeaders.CONTENT_TYPE).orElse(null);
    }

    /**
     * 把上游响应体 publisher 适配为 InputStream：逐块交付（保留上游分块边界，供 SSE 逐事件 flush），
     * 空闲超时后视为上游卡死抛 IOException；close() 取消上游订阅。
     */
    private static final class UpstreamStream extends InputStream implements Flow.Subscriber<List<ByteBuffer>> {

        private static final int QUEUE_CAPACITY = 16;

        private final BlockingQueue<byte[]> chunks = new LinkedBlockingQueue<>(QUEUE_CAPACITY);
        private final long idleTimeoutMs;

        private volatile Flow.Subscription subscription;
        private volatile Throwable error;
        private volatile boolean completed;
        private volatile boolean closed;

        private byte[] current;
        private int offset;

        private UpstreamStream(long idleTimeoutMs) {
            this.idleTimeoutMs = idleTimeoutMs;
        }

        @Override
        public void onSubscribe(Flow.Subscription subscription) {
            this.subscription = subscription;
            subscription.request(1);
        }

        @Override
        public void onNext(List<ByteBuffer> buffers) {
            int size = buffers.stream().mapToInt(ByteBuffer::remaining).sum();
            if (size > 0) {
                byte[] chunk = new byte[size];
                int position = 0;
                for (ByteBuffer buffer : buffers) {
                    int length = buffer.remaining();
                    buffer.get(chunk, position, length);
                    position += length;
                }
                try {
                    chunks.put(chunk);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    cancel();
                    return;
                }
            }
            // 背压：下游取走一块后才向上游要下一块
            Flow.Subscription subscription = this.subscription;
            if (subscription != null) {
                subscription.request(1);
            }
        }

        @Override
        public void onError(Throwable throwable) {
            this.error = throwable;
            this.completed = true;
            wakeUp();
        }

        @Override
        public void onComplete() {
            this.completed = true;
            wakeUp();
        }

        /**
         * 唤醒阻塞在队列上的下游读取线程。仅置 completed 标记无法唤醒已进入 poll(timeout) 的线程，
         * 会让"上游已结束但无后续数据块"的流（如过期 streamSessionId 重连）空等到空闲超时才收尾。
         */
        private void wakeUp() {
            chunks.offer(new byte[0]);
        }

        @Override
        public int read() throws IOException {
            byte[] single = new byte[1];
            int read = read(single, 0, 1);
            return read == -1 ? -1 : single[0] & 0xFF;
        }

        @Override
        public int read(byte[] buffer, int off, int len) throws IOException {
            if (len == 0) {
                return 0;
            }
            if (current == null && !fill()) {
                return -1;
            }
            int read = Math.min(len, current.length - offset);
            System.arraycopy(current, offset, buffer, off, read);
            offset += read;
            if (offset == current.length) {
                current = null;
                offset = 0;
            }
            return read;
        }

        @Override
        public void close() {
            closed = true;
            cancel();
        }

        private boolean fill() throws IOException {
            while (true) {
                if (closed) {
                    throw new IOException("上游响应流已关闭");
                }
                byte[] chunk = chunks.poll();
                if (chunk == null) {
                    if (error != null) {
                        throw new IOException("上游响应流异常: " + error.getMessage(), error);
                    }
                    if (completed) {
                        return false;
                    }
                    try {
                        chunk = chunks.poll(idleTimeoutMs, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new IOException("等待上游数据被中断", e);
                    }
                    if (chunk == null) {
                        if (error != null) {
                            throw new IOException("上游响应流异常: " + error.getMessage(), error);
                        }
                        if (completed) {
                            return false;
                        }
                        throw new IOException("上游响应流空闲超时(" + idleTimeoutMs + "ms)");
                    }
                }
                if (chunk.length == 0) {
                    continue;
                }
                current = chunk;
                offset = 0;
                return true;
            }
        }

        private void cancel() {
            Flow.Subscription subscription = this.subscription;
            if (subscription != null) {
                subscription.cancel();
            }
        }
    }
}
