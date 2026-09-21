package com.pei.dehaze.service.client;

import com.pei.dehaze.config.property.AlgorithmProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.AbstractWebSocketHandler;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 语音流式 ASR 的 WebSocket 转发（{@code /ws/asr} → dehaze-python）。
 * <p>
 * ASR 引擎（FunASR）与识别编排只在 python 进程内，java 只做帧搬运：不解析应用层协议，
 * 上行 PCM（二进制）与 EOS（文本）原样转发，下行识别结果 JSON 原样回传。
 * 鉴权事实源同样在 python——浏览器 WS 握手无法携带自定义头，登录会话凭证随 query 的 sid 传递，
 * python 侧校验 sid 与 ASR 会话归属；因此本端点不参与 java 的 session 鉴权
 * （见 {@code VoiceAsrWebSocketConfig} 与 {@code SecurityConfig} 的放行清单）。
 *
 * @author earthyzinc
 * @since 2026-09-18
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class VoiceAsrProxyHandler extends AbstractWebSocketHandler {

    /** 转发到 python 的握手头：会话凭证（cookie 兜底）、链路头 */
    private static final List<String> FORWARDED_HEADERS = List.of(
            HttpHeaders.COOKIE,
            "X-Session-Id",
            HttpHeaders.USER_AGENT,
            "X-Forwarded-For",
            "X-Trace-Id");

    private final AlgorithmProperties props;
    private final HttpClient aiProxyHttpClient;

    private final Map<String, Relay> relays = new ConcurrentHashMap<>();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        Relay relay = new Relay(session);
        relays.put(session.getId(), relay);
        relay.start();
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) {
        Relay relay = relays.get(session.getId());
        if (relay != null) {
            relay.forwardText(message.getPayload());
        }
    }

    @Override
    protected void handleBinaryMessage(WebSocketSession session, BinaryMessage message) {
        Relay relay = relays.get(session.getId());
        if (relay != null) {
            relay.forwardBinary(message.getPayload());
        }
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) {
        log.warn("语音 ASR WebSocket 传输异常 sessionId={}", session.getId(), exception);
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        Relay relay = relays.remove(session.getId());
        if (relay != null) {
            relay.stop();
        }
    }

    /**
     * 单条连接的转发器。
     * <p>
     * Spring 的 WebSocket 会话在下游握手完成后才交给本类，因此下游必然先于上游就绪；
     * 期间的帧按原顺序缓存，上游连上后统一重放——否则 SDK 在 onOpen 立刻发送的 PCM/EOS 会丢失，
     * 或乱序到达 python 导致识别结果错位。
     */
    private final class Relay {

        private final WebSocketSession downstream;
        private final Object lock = new Object();
        private final Deque<Runnable> pending = new ArrayDeque<>();

        private WebSocket upstream;
        private boolean closed;

        private Relay(WebSocketSession downstream) {
            this.downstream = downstream;
        }

        /** 异步连接上游：python 收到握手后要先加载 ASR 引擎（首次 20~40s）才 accept，不能阻塞容器线程 */
        private void start() {
            WebSocket.Builder builder = aiProxyHttpClient.newWebSocketBuilder();
            HttpHeaders handshakeHeaders = downstream.getHandshakeHeaders();
            for (String name : FORWARDED_HEADERS) {
                List<String> values = handshakeHeaders.get(name);
                if (values != null) {
                    values.forEach(value -> builder.header(name, value));
                }
            }
            // 刻意不设 connectTimeout：该超时约束的是握手完成，而 python 侧首次连接要等引擎冷加载
            builder.buildAsync(upstreamUri(), new UpstreamListener(this))
                    .whenComplete((webSocket, error) -> {
                        synchronized (lock) {
                            if (error != null) {
                                log.error("语音 ASR 上游连接失败 sessionId={}", downstream.getId(), error);
                                closed = true;
                                pending.clear();
                                closeDownstream(new CloseStatus(1011, "识别服务不可用"));
                                return;
                            }
                            // 上游握手期间下游已断开：直接关掉刚建立的上游连接，不留悬挂会话
                            if (closed) {
                                webSocket.sendClose(WebSocket.NORMAL_CLOSURE, "client closed");
                                return;
                            }
                            upstream = webSocket;
                            pending.forEach(Runnable::run);
                            pending.clear();
                        }
                    });
        }

        private URI upstreamUri() {
            URI request = downstream.getUri();
            String base = props.getBaseUrl();
            String wsBase = base.startsWith("https://")
                    ? "wss://" + base.substring("https://".length())
                    : base.replaceFirst("^http://", "ws://");
            String query = request.getQuery();
            return URI.create(wsBase + request.getPath() + (query == null ? "" : "?" + query));
        }

        private void forwardText(String payload) {
            forward(true, payload.getBytes(StandardCharsets.UTF_8));
        }

        private void forwardBinary(ByteBuffer payload) {
            byte[] bytes = new byte[payload.remaining()];
            payload.get(bytes);
            forward(false, bytes);
        }

        private void forward(boolean text, byte[] payload) {
            synchronized (lock) {
                if (closed) {
                    return;
                }
                if (upstream == null) {
                    pending.add(() -> send(text, payload));
                    return;
                }
                send(text, payload);
            }
        }

        /** 上游异步发送，同一连接上的调用顺序即投递顺序（由 lock 串行化） */
        private void send(boolean text, byte[] payload) {
            if (text) {
                upstream.sendText(new String(payload, StandardCharsets.UTF_8), true);
            } else {
                upstream.sendBinary(ByteBuffer.wrap(payload), true);
            }
        }

        private void sendDownstreamText(String payload) {
            writeDownstream(new TextMessage(payload));
        }

        private void sendDownstreamBinary(byte[] payload) {
            writeDownstream(new BinaryMessage(payload));
        }

        /** 只有上游监听线程会写下游会话，无需额外串行化 */
        private void writeDownstream(org.springframework.web.socket.WebSocketMessage<?> message) {
            try {
                if (downstream.isOpen()) {
                    downstream.sendMessage(message);
                }
            } catch (IOException e) {
                log.warn("语音 ASR 下游发送失败 sessionId={}", downstream.getId(), e);
                stop();
            }
        }

        private void onUpstreamClosed(CloseStatus status) {
            synchronized (lock) {
                closed = true;
                pending.clear();
            }
            closeDownstream(status);
        }

        private void onUpstreamError(Throwable error) {
            log.warn("语音 ASR 上游异常 sessionId={}", downstream.getId(), error);
            synchronized (lock) {
                closed = true;
                pending.clear();
            }
            closeDownstream(new CloseStatus(1011, "ASR 服务异常"));
        }

        private void stop() {
            synchronized (lock) {
                if (closed) {
                    return;
                }
                closed = true;
                pending.clear();
                if (upstream != null) {
                    upstream.sendClose(WebSocket.NORMAL_CLOSURE, "client closed");
                }
            }
        }

        private void closeDownstream(CloseStatus status) {
            try {
                if (downstream.isOpen()) {
                    downstream.close(status);
                }
            } catch (IOException e) {
                log.debug("关闭下游 WebSocket 失败 sessionId={}", downstream.getId(), e);
            }
        }
    }

    /** 上游 → 下游：识别结果 JSON 文本；上游关闭/异常时同步收敛下游 */
    private static final class UpstreamListener implements WebSocket.Listener {

        private final Relay relay;
        private final ByteArrayOutputStream partialText = new ByteArrayOutputStream();
        private final ByteArrayOutputStream partialBinary = new ByteArrayOutputStream();

        private UpstreamListener(Relay relay) {
            this.relay = relay;
        }

        @Override
        public void onOpen(WebSocket webSocket) {
            webSocket.request(1);
        }

        @Override
        public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
            partialText.writeBytes(data.toString().getBytes(StandardCharsets.UTF_8));
            if (last) {
                relay.sendDownstreamText(partialText.toString(StandardCharsets.UTF_8));
                partialText.reset();
            }
            webSocket.request(1);
            return null;
        }

        @Override
        public CompletionStage<?> onBinary(WebSocket webSocket, ByteBuffer data, boolean last) {
            byte[] bytes = new byte[data.remaining()];
            data.get(bytes);
            partialBinary.writeBytes(bytes);
            if (last) {
                relay.sendDownstreamBinary(partialBinary.toByteArray());
                partialBinary.reset();
            }
            webSocket.request(1);
            return null;
        }

        @Override
        public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
            relay.onUpstreamClosed(new CloseStatus(statusCode, reason));
            return null;
        }

        @Override
        public void onError(WebSocket webSocket, Throwable error) {
            relay.onUpstreamError(error);
        }
    }
}
