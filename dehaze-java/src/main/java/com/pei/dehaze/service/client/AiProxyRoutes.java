package com.pei.dehaze.service.client;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.util.AntPathMatcher;

import java.util.List;
import java.util.Optional;

/**
 * 转发白名单（python 唯一实现的端点：AI 域 B 类——强依赖 deepagents / LLM / 向量检索，
 * 以及语音域——ASR/TTS 引擎为 python 进程内本地引擎）。
 * <p>
 * 这些端点的行为事实源是 dehaze-python，java 只做 controller 层转发。白名单是转发面唯一入口：
 * 未登记的路径一律 404，绝不按客户端传入的路径盲转发（防 SSRF）。
 * 清单必须与 dehaze-python/app/router 下对应路由逐条对齐，增删接口时同步维护。
 * <p>
 * 仅覆盖 HTTP 端点；语音流式 ASR 的 WebSocket（{@code /ws/asr}）由
 * {@link com.pei.dehaze.config.VoiceWebSocketProxyConfig} 单独转发。
 *
 * @author earthyzinc
 * @since 2026-09-17
 */
public final class AiProxyRoutes {

    /** 端点性质：决定请求体与响应体的转发方式 */
    public enum Kind {
        /** SSE 长连接：需流式管道转发，客户端断连时取消上游订阅 */
        SSE,
        /** multipart 原始字节流：跳过 Spring 解析，保留文件字段名与 boundary 原样转发 */
        MULTIPART,
        /** 普通 JSON：整体读取后透传 */
        JSON
    }

    /**
     * @param authExempt 是否豁免 java 侧鉴权：第三方协议端点的认证头形态（OpenAI Bearer / Claude x-api-key）
     *                   java 过滤器不识别，须放行到 python 由 API Key 中间件终审
     */
    public record Route(String method, String pattern, Kind kind, boolean authExempt) {
    }

    private static final AntPathMatcher MATCHER = new AntPathMatcher();

    private static final String GET = "GET";
    private static final String POST = "POST";
    private static final String PUT = "PUT";
    private static final String DELETE = "DELETE";

    private static final List<Route> ROUTES = List.of(
            // ==================== AI 对话（SSE 流式推理） ====================
            sse(POST, "/api/v1/ai/conversations/{convId}/messages"),
            sse(GET, "/api/v1/ai/conversations/{convId}/messages/stream/{streamSessionId}"),
            sse(POST, "/api/v1/ai/messages/{msgId}/regenerate"),
            sse(POST, "/api/v1/ai/messages/{msgId}/resume"),
            sse(PUT, "/api/v1/ai/messages/{msgId}"),
            json(POST, "/api/v1/ai/messages/{msgId}/stop"),

            // ==================== 智能体 / Skill / 模型 / 供应商 试运行 ====================
            json(POST, "/api/v1/ai/agents/{agentId}/test"),
            multipart(POST, "/api/v1/ai/skills/upload"),
            json(POST, "/api/v1/ai/skills/{skillId}/test"),
            json(POST, "/api/v1/ai/models/{modelId}/test"),
            json(POST, "/api/v1/ai/providers/{providerId}/test-connection"),
            // 供应商熔断状态由 python 进程内维护，解除熔断属于运行时操作而非配置 CRUD
            json(POST, "/api/v1/ai/providers/{providerId}/circuit/close"),

            // ==================== 外部 MCP Server 探测 ====================
            json(GET, "/api/v1/ai/mcp/servers/{serverId}/health"),
            json(GET, "/api/v1/ai/mcp/servers/{serverId}/tools"),
            json(POST, "/api/v1/ai/mcp/servers/{serverId}/tools/test"),

            // ==================== 智能体评测（异步任务） ====================
            json(POST, "/api/v1/ai/agents/{agentId}/eval/runs"),
            json(GET, "/api/v1/ai/agents/{agentId}/eval/tasks/{taskId}"),

            // 发布门禁会真实执行回归评测（LLM 依赖），行为事实源在 python；java 原生实现无法等价
            json(POST, "/api/v1/ai/agents/{agentId}/publish"),

            // ==================== 定时任务手动触发 ====================
            json(POST, "/api/v1/ai/scheduled-tasks/{scheduleId}/run"),

            // ==================== 知识库（库级 ES 索引生命周期 / 文档处理链路 / 向量检索） ====================
            // 库创建/删除/索引统计依赖 ES 索引生命周期（python ensure_kb_index / delete_kb_index / _stats），
            // java 原生实现无法保证"库建了索引不存在"，按归属裁决一律转发
            json(POST, "/api/v1/kb"),
            json(GET, "/api/v1/kb/{kbId}/index-stats"),
            json(DELETE, "/api/v1/kb/{kbId}"),
            json(POST, "/api/v1/kb/{kbId}/documents"),
            json(POST, "/api/v1/kb/{kbId}/documents/batch"),
            json(POST, "/api/v1/kb/{kbId}/documents/import-url"),
            json(POST, "/api/v1/kb/{kbId}/documents/text"),
            json(POST, "/api/v1/kb/documents/{documentId}/reprocess"),
            // 文档版本更新/删除必须驱动 python 的 _process_document_guarded（版本快照 + 重建向量 + 状态机），
            // java 原生实现不驱动该链路，落库后 processingStatus 会永久停在 pending，故一律转发
            json(PUT, "/api/v1/kb/documents/{documentId}"),
            json(DELETE, "/api/v1/kb/documents/{documentId}"),
            json(POST, "/api/v1/kb/documents/chunks/preview"),
            json(POST, "/api/v1/kb/search"),
            json(POST, "/api/v1/kb/{kbId}/retrieve/test"),
            json(POST, "/api/v1/kb/{kbId}/retrieve/test-sets/{testSetId}/run"),

            // ==================== 语音域（ASR/TTS 引擎仅在 python 进程内） ====================
            // FunASR / Piper 为 python 进程内懒加载的本地引擎（模型随 python 分发），
            // java 无等价实现；权限（voice:hotword:edit / voice:service:monitor / voice:engine:manage）
            // 与 TTS 缓存音频的归属校验同样只在 python，故整域转发
            json(POST, "/api/v1/voice/asr/stream-session"),
            json(GET, "/api/v1/voice/asr/result/{sessionId}"),
            // 离线识别为 multipart 直传音频文件，不能交由 Spring 解析（见 AiProxyMultipartResolver）
            multipart(POST, "/api/v1/voice/asr/offline"),
            json(POST, "/api/v1/voice/tts"),
            json(GET, "/api/v1/voice/tts/voices"),
            // 缓存音频下载：响应体是解密后的音频二进制，原样回传
            json(GET, "/api/v1/voice/tts/audio/{cacheKey}"),
            json(GET, "/api/v1/voice/hotwords"),
            json(POST, "/api/v1/voice/hotwords"),
            json(DELETE, "/api/v1/voice/hotwords/{hotwordId}"),
            json(GET, "/api/v1/voice/hotwords/global"),
            json(POST, "/api/v1/voice/hotwords/global"),
            json(DELETE, "/api/v1/voice/hotwords/global/{hotwordId}"),
            json(GET, "/api/v1/voice/service/status"),
            json(GET, "/api/v1/voice/providers"),
            json(GET, "/api/v1/voice/providers/enabled"),
            json(POST, "/api/v1/voice/providers"),
            json(PUT, "/api/v1/voice/providers/{providerId}"),
            json(DELETE, "/api/v1/voice/providers/{providerId}"),
            // 连通性测试要真实访问引擎（云端引擎走外网、local 引擎走进程内实例）
            json(POST, "/api/v1/voice/providers/{providerId}/test-connection"),
            json(GET, "/api/v1/voice/providers/{providerId}/keys"),
            json(POST, "/api/v1/voice/providers/{providerId}/keys"),
            json(PUT, "/api/v1/voice/providers/{providerId}/keys/{keyId}"),
            json(DELETE, "/api/v1/voice/providers/{providerId}/keys/{keyId}"),
            json(GET, "/api/v1/voice/models"),
            json(POST, "/api/v1/voice/models"),
            json(PUT, "/api/v1/voice/models/{modelId}"),
            json(DELETE, "/api/v1/voice/models/{modelId}"),

            // ==================== A2A 协议（message/stream 为 SSE） ====================
            // 挂载路径下的 Agent Card 由 java 原生实现（AiA2aController 按已发布版本动态生成），
            // 此处只转发 JSON-RPC 入口与全局发现端点
            sse(POST, "/api/v1/ai/agents/{agentId}/a2a"),
            json(POST, "/api/v1/ai/a2a/endpoints/{endpointId}/refresh-card"),

            // ==================== OpenAI / Claude 兼容协议（第三方 SDK 接入，鉴权交 python 终审） ====================
            authExempt(sse(POST, "/api/v1/chat/completions")),
            authExempt(json(GET, "/api/v1/models")),
            authExempt(sse(POST, "/api/v1/messages")),
            authExempt(sse(POST, "/a2a")),
            authExempt(json(GET, "/.well-known/agent.json"))
    );

    private static final List<Route> AUTH_EXEMPT_ROUTES = ROUTES.stream().filter(Route::authExempt).toList();

    private AiProxyRoutes() {
    }

    private static Route sse(String method, String pattern) {
        return new Route(method, pattern, Kind.SSE, false);
    }

    private static Route json(String method, String pattern) {
        return new Route(method, pattern, Kind.JSON, false);
    }

    private static Route multipart(String method, String pattern) {
        return new Route(method, pattern, Kind.MULTIPART, false);
    }

    private static Route authExempt(Route route) {
        return new Route(route.method(), route.pattern(), route.kind(), true);
    }

    /** 请求路径（不含 query），与白名单模板同一口径 */
    public static String pathOf(HttpServletRequest request) {
        String uri = request.getRequestURI();
        String contextPath = request.getContextPath();
        return contextPath == null || contextPath.isEmpty() ? uri : uri.substring(contextPath.length());
    }

    /** 查询端点性质；未登记（白名单外）返回空 */
    public static Optional<Kind> kindOf(String method, String path) {
        for (Route route : ROUTES) {
            if (route.method().equalsIgnoreCase(method) && MATCHER.match(route.pattern(), path)) {
                return Optional.of(route.kind());
            }
        }
        return Optional.empty();
    }

    public static boolean isRawMultipart(String method, String path) {
        return kindOf(method, path).filter(Kind.MULTIPART::equals).isPresent();
    }

    /** 是否豁免 java 侧鉴权（认证头形态由 python 终审的第三方协议端点） */
    public static boolean isAuthExempt(String method, String path) {
        return AUTH_EXEMPT_ROUTES.stream()
                .anyMatch(route -> route.method().equalsIgnoreCase(method) && MATCHER.match(route.pattern(), path));
    }

    /** 供安全过滤器链注册 permitAll 使用，保证豁免清单只有一处定义 */
    public static List<Route> authExemptRoutes() {
        return AUTH_EXEMPT_ROUTES;
    }
}
