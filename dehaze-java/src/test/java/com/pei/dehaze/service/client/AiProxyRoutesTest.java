package com.pei.dehaze.service.client;

import com.pei.dehaze.service.client.AiProxyRoutes.Kind;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * AI 转发白名单单元测试：端点性质登记与路径匹配边界。
 */
@DisplayName("AiProxyRoutes 白名单单元测试")
class AiProxyRoutesTest {

    @ParameterizedTest(name = "{0} {1} -> {2}")
    @DisplayName("已登记端点按性质匹配，路径参数正常命中")
    @CsvSource({
            "POST,   /api/v1/ai/conversations/12/messages,                              SSE",
            "GET,    /api/v1/ai/conversations/12/messages/stream/abc-123,               SSE",
            "POST,   /api/v1/ai/messages/34/regenerate,                                 SSE",
            "POST,   /api/v1/ai/messages/34/resume,                                     SSE",
            "PUT,    /api/v1/ai/messages/34,                                            SSE",
            "POST,   /api/v1/ai/messages/34/stop,                                       JSON",
            "POST,   /api/v1/ai/agents/5/test,                                          JSON",
            "POST,   /api/v1/ai/skills/upload,                                          MULTIPART",
            "POST,   /api/v1/ai/skills/6/test,                                          JSON",
            "POST,   /api/v1/ai/models/gpt-4o/test,                                     JSON",
            "POST,   /api/v1/ai/providers/3/test-connection,                            JSON",
            "POST,   /api/v1/ai/providers/3/circuit/close,                              JSON",
            "GET,    /api/v1/ai/mcp/servers/7/health,                                   JSON",
            "GET,    /api/v1/ai/mcp/servers/7/tools,                                    JSON",
            "POST,   /api/v1/ai/mcp/servers/7/tools/test,                               JSON",
            "POST,   /api/v1/ai/agents/5/eval/runs,                                     JSON",
            "GET,    /api/v1/ai/agents/5/eval/tasks/task-1,                             JSON",
            "POST,   /api/v1/ai/agents/5/publish,                                       JSON",
            "POST,   /api/v1/ai/scheduled-tasks/9/run,                                  JSON",
            "POST,   /api/v1/kb,                                                       JSON",
            "GET,    /api/v1/kb/2/index-stats,                                         JSON",
            "DELETE, /api/v1/kb/2,                                                     JSON",
            "POST,   /api/v1/kb/2/documents,                                            JSON",
            "POST,   /api/v1/kb/2/documents/batch,                                      JSON",
            "POST,   /api/v1/kb/2/documents/import-url,                                 JSON",
            "POST,   /api/v1/kb/2/documents/text,                                       JSON",
            "POST,   /api/v1/kb/documents/88/reprocess,                                 JSON",
            "PUT,    /api/v1/kb/documents/88,                                           JSON",
            "DELETE, /api/v1/kb/documents/88,                                           JSON",
            "POST,   /api/v1/kb/documents/chunks/preview,                               JSON",
            "POST,   /api/v1/kb/search,                                                 JSON",
            "POST,   /api/v1/kb/2/retrieve/test,                                        JSON",
            "POST,   /api/v1/kb/2/retrieve/test-sets/3/run,                             JSON",
            "POST,   /api/v1/ai/agents/5/a2a,                                           SSE",
            "POST,   /api/v1/ai/a2a/endpoints/4/refresh-card,                           JSON",
            "POST,   /a2a,                                                              SSE",
            "GET,    /.well-known/agent.json,                                           JSON",
            "POST,   /api/v1/chat/completions,                                          SSE",
            "GET,    /api/v1/models,                                                    JSON",
            "POST,   /api/v1/messages,                                                  SSE",
            // 语音域（引擎仅在 python 进程内）
            "POST,   /api/v1/voice/asr/stream-session,                                  JSON",
            "GET,    /api/v1/voice/asr/result/abc-123,                                  JSON",
            "POST,   /api/v1/voice/asr/offline,                                         MULTIPART",
            "POST,   /api/v1/voice/tts,                                                 JSON",
            "GET,    /api/v1/voice/tts/voices,                                          JSON",
            "GET,    /api/v1/voice/tts/audio/abc123,                                    JSON",
            "GET,    /api/v1/voice/hotwords,                                            JSON",
            "POST,   /api/v1/voice/hotwords,                                            JSON",
            "DELETE, /api/v1/voice/hotwords/9,                                          JSON",
            "GET,    /api/v1/voice/hotwords/global,                                     JSON",
            "POST,   /api/v1/voice/hotwords/global,                                     JSON",
            "DELETE, /api/v1/voice/hotwords/global/9,                                   JSON",
            "GET,    /api/v1/voice/service/status,                                      JSON",
            "GET,    /api/v1/voice/providers,                                           JSON",
            "GET,    /api/v1/voice/providers/enabled,                                   JSON",
            "POST,   /api/v1/voice/providers,                                           JSON",
            "PUT,    /api/v1/voice/providers/3,                                         JSON",
            "DELETE, /api/v1/voice/providers/3,                                         JSON",
            "POST,   /api/v1/voice/providers/3/test-connection,                         JSON",
            "GET,    /api/v1/voice/providers/3/keys,                                    JSON",
            "POST,   /api/v1/voice/providers/3/keys,                                    JSON",
            "PUT,    /api/v1/voice/providers/3/keys/5,                                  JSON",
            "DELETE, /api/v1/voice/providers/3/keys/5,                                  JSON",
            "GET,    /api/v1/voice/models,                                              JSON",
            "POST,   /api/v1/voice/models,                                              JSON",
            "PUT,    /api/v1/voice/models/5,                                            JSON",
            "DELETE, /api/v1/voice/models/5,                                            JSON"
    })
    void kindOf_matchesRegisteredRoutes(String method, String path, Kind expected) {
        assertThat(AiProxyRoutes.kindOf(method.trim(), path.trim())).contains(expected);
    }

    @Test
    @DisplayName("白名单外路径返回空（禁止按客户端路径盲转发）")
    void kindOf_rejectsUnknownPaths() {
        assertThat(AiProxyRoutes.kindOf("GET", "/api/v1/ai/agents")).isEmpty();
        assertThat(AiProxyRoutes.kindOf("GET", "/api/v1/orders")).isEmpty();
        assertThat(AiProxyRoutes.kindOf("GET", "/internal/admin")).isEmpty();
        assertThat(AiProxyRoutes.kindOf("POST", "/api/v1/ai/skills/upload/extra")).isEmpty();
        // 语音流式 ASR 的 WebSocket 不在 HTTP 白名单内：它由 VoiceAsrWebSocketConfig 单独转发，
        // 混进白名单只会让 controller 收到一个无法升级的 GET
        assertThat(AiProxyRoutes.kindOf("GET", "/ws/asr")).isEmpty();
    }

    @Test
    @DisplayName("方法不在白名单内同样拒绝（A 类 CRUD 端点不由转发承接）")
    void kindOf_rejectsUnregisteredMethods() {
        // 会话消息列表是 A 类 CRUD，仅发送(POST)走转发
        assertThat(AiProxyRoutes.kindOf("GET", "/api/v1/ai/conversations/12/messages")).isEmpty();
        // 消息删除是 A 类 CRUD，仅编辑(PUT)走转发
        assertThat(AiProxyRoutes.kindOf("DELETE", "/api/v1/ai/messages/34")).isEmpty();
        assertThat(AiProxyRoutes.kindOf("GET", "/api/v1/ai/skills/upload")).isEmpty();
    }

    @Test
    @DisplayName("isRawMultipart 仅对 multipart 端点成立")
    void isRawMultipart_onlyForMultipartRoutes() {
        assertThat(AiProxyRoutes.isRawMultipart("POST", "/api/v1/ai/skills/upload")).isTrue();
        // 离线 ASR 直传音频，必须跳过 Spring 解析保留 boundary 与文件流
        assertThat(AiProxyRoutes.isRawMultipart("POST", "/api/v1/voice/asr/offline")).isTrue();
        assertThat(AiProxyRoutes.isRawMultipart("GET", "/api/v1/voice/tts/voices")).isFalse();
        assertThat(AiProxyRoutes.isRawMultipart("POST", "/api/v1/kb/2/documents")).isFalse();
        assertThat(AiProxyRoutes.isRawMultipart("POST", "/api/v1/files/upload")).isFalse();
    }

    @Test
    @DisplayName("鉴权豁免仅覆盖第三方协议端点与 A2A 全局入口")
    void isAuthExempt_coversThirdPartyProtocolEndpointsOnly() {
        assertThat(AiProxyRoutes.authExemptRoutes()).hasSize(5);
        assertThat(AiProxyRoutes.isAuthExempt("POST", "/api/v1/chat/completions")).isTrue();
        assertThat(AiProxyRoutes.isAuthExempt("POST", "/api/v1/messages")).isTrue();
        assertThat(AiProxyRoutes.isAuthExempt("GET", "/api/v1/models")).isTrue();
        assertThat(AiProxyRoutes.isAuthExempt("POST", "/a2a")).isTrue();
        assertThat(AiProxyRoutes.isAuthExempt("GET", "/.well-known/agent.json")).isTrue();

        // 内部会话端点与 A 类端点仍需本地鉴权，不能被豁免清单误放行
        assertThat(AiProxyRoutes.isAuthExempt("GET", "/api/v1/messages")).isFalse();
        assertThat(AiProxyRoutes.isAuthExempt("POST", "/api/v1/ai/conversations/1/messages")).isFalse();
        assertThat(AiProxyRoutes.isAuthExempt("POST", "/api/v1/ai/skills/upload")).isFalse();
        assertThat(AiProxyRoutes.isAuthExempt("GET", "/api/v1/kb/search")).isFalse();
        // 文档变更走常规 session 鉴权（与其它 KB 转发一致），不得误入豁免清单
        assertThat(AiProxyRoutes.isAuthExempt("PUT", "/api/v1/kb/documents/88")).isFalse();
        assertThat(AiProxyRoutes.isAuthExempt("DELETE", "/api/v1/kb/documents/88")).isFalse();
    }

    @Test
    @DisplayName("挂载路径 Agent Card 归 java 原生实现，不在转发白名单内")
    void mountedAgentCard_isNotForwarded() {
        assertThat(AiProxyRoutes.kindOf("GET", "/api/v1/ai/agents/5/a2a/.well-known/agent.json")).isEmpty();
    }

    @Test
    @DisplayName("鉴权豁免清单与转发白名单同源：豁免端点必可转发")
    void authExemptRoutes_areAllForwardable() {
        AiProxyRoutes.authExemptRoutes().forEach(route ->
                assertThat(AiProxyRoutes.kindOf(route.method(), route.pattern())).isPresent());
    }
}
