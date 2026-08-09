package com.pei.dehaze.service.client;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.config.property.AlgorithmProperties;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.micrometer.core.instrument.Timer;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.net.HttpURLConnection;
import java.net.URI;
import java.util.UUID;

/**
 * Python 算法服务 HTTP 客户端 — 生产级实现
 * <p>
 * 特性：
 * <ul>
 *   <li>指数退避重试（最大3次）</li>
 *   <li>Resilience4j 熔断器（滑动窗口统计、半开探活）</li>
 *   <li>统一 JSON 响应解析</li>
 *   <li>区分可重试/不可重试错误</li>
 * </ul>
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class PythonAlgorithmClient {

    private final AlgorithmProperties props;
    private final RestTemplate algorithmRestTemplate;
    private final CircuitBreaker circuitBreaker;
    private final Timer pythonCallTimer;

    @PostConstruct
    public void init() {
        log.info("PythonAlgorithmClient 初始化完成: baseUrl={}, connectTimeout={}ms, readTimeout={}ms, maxRetry={}",
                props.getBaseUrl(), props.getConnectTimeout(), props.getReadTimeout(), props.getMaxRetry());
    }

    // ==================== 公开 API ====================

    /**
     * 调用预测服务：发送雾图 → 返回去雾结果
     */
    public JSONObject predict(Long algorithmId, String imageUrl, String params) {
        JSONObject body = new JSONObject();
        body.set("algorithmId", algorithmId);
        body.set("imageUrl", imageUrl);
        if (params != null) {
            // Python 端约定 params 为 JSON 字符串，直接透传，不能解析成对象
            body.set("params", params);
        }
        return pythonCallTimer.record(() -> postWithRetry(props.getPredictPath(), body.toString()));
    }

    /**
     * 调用评估服务：发送预测图+参考图 → 返回评估指标
     */
    public JSONObject evaluate(Long algorithmId, String predUrl, String gtUrl) {
        JSONObject body = new JSONObject();
        body.set("algorithmId", algorithmId);
        body.set("predUrl", predUrl);
        body.set("gtUrl", gtUrl);
        return pythonCallTimer.record(() -> postWithRetry(props.getEvaluatePath(), body.toString()));
    }

    /**
     * 调用图像特征分析服务：发送图像 URL → 返回 7 维结构化特征（供推荐模块使用）
     * <p>
     * Python 服务不可用时由 postWithRetry 抛出 BusinessException，不降级为伪特征，避免误导用户。
     */
    public JSONObject analyzeImage(String imageUrl) {
        JSONObject body = new JSONObject();
        body.set("imageUrl", imageUrl);
        return pythonCallTimer.record(() -> postWithRetry(props.getAnalyzePath(), body.toString()));
    }

    /**
     * 查询预测任务状态（用于异步轮询）
     */
    public JSONObject getPredTaskStatus(Long taskId) {
        return getWithRetry(props.getPredictPath() + "/" + taskId);
    }

    /**
     * 查询评估任务状态（用于异步轮询）
     */
    public JSONObject getEvalTaskStatus(Long taskId) {
        return getWithRetry(props.getEvaluatePath() + "/" + taskId);
    }

    // ==================== 核心请求方法 ====================

    private JSONObject postWithRetry(String path, String jsonBody) {
        String url = props.getBaseUrl() + path;

        // 生成幂等键：同一逻辑请求的所有重试共用同一个键，便于下游去重
        String idempotencyKey = UUID.randomUUID().toString();

        Exception lastException = null;
        long backoff = props.getRetryBackoff();

        for (int attempt = 0; attempt <= props.getMaxRetry(); attempt++) {
            try {
                if (attempt > 0) {
                    log.debug("重试第 {} 次: {} (退避 {}ms)", attempt, url, backoff);
                    Thread.sleep(backoff);
                    backoff *= 2; // 指数退避
                }

                // 每次 doPost 交由 Resilience4j 熔断器统计成功/失败
                return circuitBreaker.executeSupplier(() -> doPost(url, jsonBody, idempotencyKey));

            } catch (CallNotPermittedException e) {
                // 熔断器开启：先探活 Python 服务，若健康则强制恢复并重试本次调用
                if (isPythonServiceHealthy()) {
                    log.warn("Python 服务健康但熔断器处于 OPEN 状态，强制恢复并重试");
                    circuitBreaker.transitionToClosedState();
                    // 恢复后跳过退避，立即重试本次调用（不计入 maxRetry）
                    try {
                        return circuitBreaker.executeSupplier(() -> doPost(url, jsonBody, idempotencyKey));
                    } catch (CallNotPermittedException ex) {
                        throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Python 算法服务熔断中，请稍后重试");
                    }
                }
                throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Python 算法服务熔断中，请稍后重试");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Python 算法服务调用被中断");
            } catch (ResourceAccessException e) {
                // 网络层异常（连接超时/拒绝）→ 可重试（熔断器已通过 executeSupplier 记录失败）
                lastException = e;
                log.warn("Python 服务网络异常 (attempt={}/{}): {}", attempt + 1, props.getMaxRetry() + 1, e.getMessage(), e);
            } catch (BusinessException e) {
                // 业务异常（Python 返回错误码）→ 不重试（熔断器已通过 executeSupplier 记录失败）
                throw e;
            } catch (Exception e) {
                lastException = e;
                log.warn("调用 Python 服务异常 (attempt={}/{}): {}", attempt + 1, props.getMaxRetry() + 1, e.getMessage(), e);
            }
        }

        // 全部重试失败
        String msg = "Python 算法服务调用失败 (已重试 " + props.getMaxRetry() + " 次): " +
                (lastException != null ? lastException.getMessage() : "未知错误");
        log.error(msg, lastException);
        throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, msg);
    }

    private JSONObject getWithRetry(String path) {
        String url = props.getBaseUrl() + path;

        Exception lastException = null;
        long backoff = props.getRetryBackoff();

        for (int attempt = 0; attempt <= props.getMaxRetry(); attempt++) {
            try {
                if (attempt > 0) {
                    log.debug("重试第 {} 次: {} (退避 {}ms)", attempt, url, backoff);
                    Thread.sleep(backoff);
                    backoff *= 2;
                }
                return circuitBreaker.executeSupplier(() -> doGet(url));
            } catch (CallNotPermittedException e) {
                // 熔断器开启：先探活 Python 服务，若健康则强制恢复并重试本次调用
                if (isPythonServiceHealthy()) {
                    log.warn("Python 服务健康但熔断器处于 OPEN 状态，强制恢复并重试");
                    circuitBreaker.transitionToClosedState();
                    try {
                        return circuitBreaker.executeSupplier(() -> doGet(url));
                    } catch (CallNotPermittedException ex) {
                        throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Python 算法服务熔断中，请稍后重试");
                    }
                }
                throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Python 算法服务熔断中，请稍后重试");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Python 算法服务调用被中断");
            } catch (ResourceAccessException e) {
                lastException = e;
                log.warn("Python 服务网络异常 (attempt={}/{}): {}", attempt + 1, props.getMaxRetry() + 1, e.getMessage(), e);
            } catch (BusinessException e) {
                throw e;
            } catch (Exception e) {
                lastException = e;
                log.warn("调用 Python 服务异常 (attempt={}/{}): {}", attempt + 1, props.getMaxRetry() + 1, e.getMessage(), e);
            }
        }

        String msg = "Python 算法服务调用失败 (已重试 " + props.getMaxRetry() + " 次): " +
                (lastException != null ? lastException.getMessage() : "未知错误");
        log.error(msg, lastException);
        throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, msg);
    }

    private HttpHeaders buildAuthHeaders(String idempotencyKey) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        if (idempotencyKey != null) {
            headers.set("X-Idempotency-Key", idempotencyKey);
        }
        String traceId = MDC.get("trace_id");
        if (traceId != null && !traceId.isBlank()) {
            headers.set("X-Trace-Id", traceId);
        }
        ServletRequestAttributes requestAttributes =
                (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (requestAttributes != null) {
            String authorization = requestAttributes.getRequest().getHeader(HttpHeaders.AUTHORIZATION);
            if (authorization != null && !authorization.isBlank()) {
                headers.set(HttpHeaders.AUTHORIZATION, authorization);
            }
        }
        if (!headers.containsKey(HttpHeaders.AUTHORIZATION) && props.getApiKey() != null && !props.getApiKey().isBlank()) {
            headers.set(HttpHeaders.AUTHORIZATION, "Bearer " + props.getApiKey());
        }
        return headers;
    }

    private JSONObject parsePythonResponse(ResponseEntity<String> response) {
        // HTTP 5xx 是服务端故障，抛 RuntimeException 让熔断器计数
        if (response.getStatusCode().is5xxServerError()) {
            throw new RuntimeException("Python 服务返回 5xx: " + response.getStatusCode() + " body=" + response.getBody());
        }
        // HTTP 4xx 是调用方问题，抛 BusinessException 不触发熔断
        if (!response.getStatusCode().is2xxSuccessful()) {
            throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Python 服务返回 4xx: " + response.getStatusCode() + " body=" + response.getBody());
        }

        String body = response.getBody();
        if (body == null || body.isBlank()) {
            throw new RuntimeException("Python 服务返回空响应");
        }

        JSONObject json = JSONUtil.parseObj(body);

        // 统一解析 Python 服务响应格式: { "code": "00000", "data": {...}, "msg": "..." }
        String code = json.getStr("code");
        if (!"00000".equals(code)) {
            String msg = json.getStr("msg", "未知错误");
            log.error("Python 服务业务异常: code={}, msg={}", code, msg);
            throw new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Python 服务错误 [" + code + "]: " + msg);
        }

        return json.getJSONObject("data");
    }

    private JSONObject doPost(String url, String jsonBody, String idempotencyKey) {
        HttpHeaders headers = buildAuthHeaders(idempotencyKey);
        HttpEntity<String> entity = new HttpEntity<>(jsonBody, headers);
        ResponseEntity<String> response = algorithmRestTemplate.postForEntity(url, entity, String.class);
        return parsePythonResponse(response);
    }

    private JSONObject doGet(String url) {
        HttpHeaders headers = buildAuthHeaders(null);
        HttpEntity<Void> entity = new HttpEntity<>(headers);
        ResponseEntity<String> response = algorithmRestTemplate.exchange(url, HttpMethod.GET, entity, String.class);
        return parsePythonResponse(response);
    }

    /**
     * 探测 Python 算法服务是否可达（轻量级 TCP/HTTP 健康检查）
     */
    private boolean isPythonServiceHealthy() {
        try {
            URI healthUri = URI.create(props.getBaseUrl() + "/health");
            HttpURLConnection conn = (HttpURLConnection) healthUri.toURL().openConnection();
            conn.setConnectTimeout(3000);
            conn.setReadTimeout(3000);
            conn.setRequestMethod("GET");
            int code = conn.getResponseCode();
            conn.disconnect();
            return code == 200;
        } catch (Exception e) {
            log.warn("Python 服务健康检查失败: {}", e.getMessage());
            return false;
        }
    }
}
