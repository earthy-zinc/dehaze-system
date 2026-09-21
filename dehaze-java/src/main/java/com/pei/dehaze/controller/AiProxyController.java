package com.pei.dehaze.controller;

import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.filter.TraceIdFilter;
import com.pei.dehaze.service.client.AiForwardClient;
import com.pei.dehaze.service.client.AiProxyRoutes;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.ServletOutputStream;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.http.HttpRequest;
import java.nio.charset.StandardCharsets;
import java.util.Enumeration;
import java.util.List;
import java.util.Optional;

import static org.springframework.web.bind.annotation.RequestMethod.DELETE;
import static org.springframework.web.bind.annotation.RequestMethod.GET;
import static org.springframework.web.bind.annotation.RequestMethod.PATCH;
import static org.springframework.web.bind.annotation.RequestMethod.POST;
import static org.springframework.web.bind.annotation.RequestMethod.PUT;

/**
 * AI 能力端点转发入口（B 类端点：deepagents / LLM / 向量检索等强依赖 Python 的能力）。
 * <p>
 * 行为事实源是 dehaze-python：java 不做业务实现，仅按 {@link AiProxyRoutes} 白名单透传请求与响应，
 * 使各前端与 SDK 在 java 端获得与 python 端一致的接口形态。
 * <p>
 * 匹配范围为通配模式，A 类 CRUD 端点由各业务 Controller 以精确路径承接（精确路径优先于通配），
 * 白名单外的路径返回 404，不做任何转发。挂载路径下的 Agent Card
 * （{@code /api/v1/ai/agents/{id}/a2a/.well-known/agent.json}）由 {@link AiA2aController}
 * 原生实现，虽落在 {@code /api/v1/**} 通配范围内，但精确映射优先命中；全局发现端点
 * {@code /.well-known/agent.json} 无原生实现，在此转发。
 *
 * @author earthyzinc
 * @since 2026-09-17
 */
@Slf4j
@Tag(name = "38.AI 能力转发")
@RestController
@RequestMapping({"/api/v1/**", "/a2a", "/.well-known/agent.json"})
@RequiredArgsConstructor
public class AiProxyController {

    /**
     * 转发头白名单：用户身份（三端共享 Redis session）、第三方协议凭据（Claude 的 x-api-key）、
     * 幂等与流式协议头。不转发 Host / Content-Length / Connection 等逐跳头，由 HttpClient 按目标连接自行生成。
     */
    private static final List<String> FORWARDED_HEADERS = List.of(
            HttpHeaders.AUTHORIZATION,
            HttpHeaders.COOKIE,
            "X-Session-Id",
            // Claude 协议的 API Key 载体，python 的 API Key 中间件据此鉴权
            "x-api-key",
            HttpHeaders.CONTENT_TYPE,
            HttpHeaders.ACCEPT,
            HttpHeaders.ACCEPT_LANGUAGE,
            HttpHeaders.CACHE_CONTROL,
            // SSE 断线重连的断点位置，必须透传
            "Last-Event-ID",
            "Idempotency-Key",
            "X-Requested-With");

    private final AiForwardClient aiForwardClient;

    @Operation(summary = "AI 能力端点转发（JSON / SSE / multipart 透传）")
    @RequestMapping(method = {GET, POST, PUT, PATCH, DELETE})
    public void forward(HttpServletRequest request, HttpServletResponse response) throws IOException, InterruptedException {
        String method = request.getMethod();
        String path = AiProxyRoutes.pathOf(request);
        Optional<AiProxyRoutes.Kind> kind = AiProxyRoutes.kindOf(method, path);
        if (kind.isEmpty()) {
            log.warn("AI 转发白名单外的请求: {} {}", method, path);
            writeJson(response, HttpStatus.NOT_FOUND.value(), Result.failed(ResultCode.RESOURCE_NOT_FOUND, "接口不存在"));
            return;
        }

        String target = path + (request.getQueryString() == null ? "" : "?" + request.getQueryString());
        HttpHeaders headers = forwardHeaders(request);
        HttpRequest.BodyPublisher body = requestBody(request, kind.get());

        try {
            if (AiProxyRoutes.Kind.SSE == kind.get()) {
                forwardStream(method, target, headers, body, response);
            } else {
                AiForwardClient.BufferedResponse upstream = aiForwardClient.exchange(method, target, headers, body);
                response.setStatus(upstream.status());
                if (upstream.contentType() != null) {
                    response.setContentType(upstream.contentType());
                }
                byte[] payload = upstream.body();
                if (payload != null && payload.length > 0) {
                    response.setContentLength(payload.length);
                    response.getOutputStream().write(payload);
                }
            }
        } catch (IOException e) {
            log.error("AI 转发失败: {} {} - {}", method, target, e.getMessage(), e);
            writeUnavailable(response);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("AI 转发被中断: {} {}", method, target);
            writeUnavailable(response);
        }
    }

    /**
     * SSE 转发：上游响应体逐块写回并 flush——servlet 缓冲会吞住事件，必须逐块刷出。
     * 客户端断连由写响应时的 IOException 暴露，据此关闭上游流（取消订阅），停止 python 侧推理。
     */
    private void forwardStream(String method, String target, HttpHeaders headers,
                               HttpRequest.BodyPublisher body, HttpServletResponse response)
            throws IOException, InterruptedException {
        aiForwardClient.stream(method, target, headers, body, (status, contentType, upstream) -> {
            response.setStatus(status);
            if (contentType != null) {
                response.setContentType(contentType);
            }
            // 与 python 端 SSE 响应一致：禁止中间层缓存事件流
            response.setHeader(HttpHeaders.CACHE_CONTROL, "no-cache");
            ServletOutputStream out = response.getOutputStream();
            byte[] buffer = new byte[4096];
            int read;
            while ((read = upstream.read(buffer)) != -1) {
                out.write(buffer, 0, read);
                out.flush();
            }
        });
    }

    /**
     * 组装转发头：透传用户会话头，python 端凭共享 session 直接鉴权并执行权限终审；同时续接链路追踪。
     */
    static HttpHeaders forwardHeaders(HttpServletRequest request) {
        HttpHeaders headers = new HttpHeaders();
        for (String name : FORWARDED_HEADERS) {
            Enumeration<String> values = request.getHeaders(name);
            while (values.hasMoreElements()) {
                headers.add(name, values.nextElement());
            }
        }
        String traceId = MDC.get(TraceIdFilter.MDC_TRACE_ID);
        if (traceId != null && !traceId.isBlank()) {
            headers.set("X-Trace-Id", traceId);
        }
        // python 侧审计与限流按客户端 IP 计（TraceMiddleware/RateLimitMiddleware 读 x-forwarded-for），
        // 取 TraceIdFilter 解析后的客户端 IP 单值转发，不透传客户端自报的整条链（防伪造）
        String clientIp = MDC.get(TraceIdFilter.MDC_IP);
        if (clientIp != null && !clientIp.isBlank()) {
            headers.set("X-Forwarded-For", clientIp);
        }
        String userAgent = MDC.get(TraceIdFilter.MDC_USER_AGENT);
        if (userAgent != null && !userAgent.isBlank()) {
            headers.set(HttpHeaders.USER_AGENT, userAgent);
        }
        return headers;
    }

    /**
     * 构建上游请求体：multipart 以原始字节流直传（不落盘、不改写 boundary）；其余端点请求体小，
     * 整体读取后转发，避免上游半途失败时向客户端输出半截响应。
     */
    private static HttpRequest.BodyPublisher requestBody(HttpServletRequest request, AiProxyRoutes.Kind kind)
            throws IOException {
        if (AiProxyRoutes.Kind.MULTIPART == kind) {
            HttpRequest.BodyPublisher raw = HttpRequest.BodyPublishers.ofInputStream(() -> {
                try {
                    return request.getInputStream();
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
            long contentLength = request.getContentLengthLong();
            return contentLength >= 0 ? HttpRequest.BodyPublishers.fromPublisher(raw, contentLength) : raw;
        }
        return HttpRequest.BodyPublishers.ofByteArray(request.getInputStream().readAllBytes());
    }

    /**
     * python 不可达时的统一降级出口；SSE 已输出首块（响应已提交）时只能断流，由客户端重连兜底。
     */
    private static void writeUnavailable(HttpServletResponse response) throws IOException {
        if (response.isCommitted()) {
            return;
        }
        writeJson(response, HttpStatus.SERVICE_UNAVAILABLE.value(),
                Result.failed(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "AI 服务暂不可用，请稍后重试"));
    }

    private static void writeJson(HttpServletResponse response, int status, Result<?> result) throws IOException {
        response.setStatus(status);
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setCharacterEncoding(StandardCharsets.UTF_8.name());
        response.getWriter().write(JSONUtil.toJsonStr(result));
        response.getWriter().flush();
    }
}
