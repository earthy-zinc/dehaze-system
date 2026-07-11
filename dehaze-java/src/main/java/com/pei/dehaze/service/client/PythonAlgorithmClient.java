package com.pei.dehaze.service.client;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.config.property.AlgorithmProperties;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/**
 * Python 算法服务 HTTP 客户端 — 生产级实现
 * <p>
 * 特性：
 * <ul>
 *   <li>指数退避重试（最大3次）</li>
 *   <li>简单熔断器（失败率超过阈值后半开探活）</li>
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

    // ---- 熔断器状态 ----
    private enum CircuitState { CLOSED, OPEN, HALF_OPEN }

    private volatile CircuitState circuitState = CircuitState.CLOSED;
    private final AtomicInteger failureCount = new AtomicInteger(0);
    private final AtomicInteger totalCount = new AtomicInteger(0);
    private final AtomicLong lastFailureTime = new AtomicLong(0);

    private final Map<String, CircuitState> endpointStates = new ConcurrentHashMap<>();
    private final Map<String, AtomicInteger> endpointFailureCounts = new ConcurrentHashMap<>();
    private final Map<String, AtomicInteger> endpointTotalCounts = new ConcurrentHashMap<>();

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
            body.set("params", JSONUtil.parseObj(params));
        }
        return postWithRetry(props.getPredictPath(), body.toString());
    }

    /**
     * 调用评估服务：发送预测图+参考图 → 返回评估指标
     */
    public JSONObject evaluate(Long algorithmId, String predUrl, String gtUrl) {
        JSONObject body = new JSONObject();
        body.set("algorithmId", algorithmId);
        body.set("predUrl", predUrl);
        body.set("gtUrl", gtUrl);
        return postWithRetry(props.getEvaluatePath(), body.toString());
    }

    // ==================== 核心请求方法 ====================

    private JSONObject postWithRetry(String path, String jsonBody) {
        String url = props.getBaseUrl() + path;

        // 熔断器检查
        checkCircuitBreaker(path);

        Exception lastException = null;
        long backoff = props.getRetryBackoff();

        for (int attempt = 0; attempt <= props.getMaxRetry(); attempt++) {
            try {
                if (attempt > 0) {
                    log.info("重试第 {} 次: {} (退避 {}ms)", attempt, url, backoff);
                    Thread.sleep(backoff);
                    backoff *= 2; // 指数退避
                }

                JSONObject result = doPost(url, jsonBody);

                // 成功后重置熔断计数器
                recordSuccess(path);
                return result;

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR.getMsg(), e);
            } catch (ResourceAccessException e) {
                // 网络层异常（连接超时/拒绝）→ 可重试
                lastException = e;
                log.warn("Python 服务网络异常 (attempt={}/{}): {}", attempt + 1, props.getMaxRetry() + 1, e.getMessage());
            } catch (BusinessException e) {
                // 业务异常（Python 返回错误码）→ 不重试
                throw e;
            } catch (Exception e) {
                lastException = e;
                log.warn("调用 Python 服务异常 (attempt={}/{}): {}", attempt + 1, props.getMaxRetry() + 1, e.getMessage());
            }
        }

        // 全部重试失败
        recordFailure(path);
        String msg = "Python 算法服务调用失败 (已重试 " + props.getMaxRetry() + " 次): " +
                (lastException != null ? lastException.getMessage() : "未知错误");
        log.error(msg);
        throw new BusinessException(msg);
    }

    private JSONObject doPost(String url, String jsonBody) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<String> entity = new HttpEntity<>(jsonBody, headers);

        ResponseEntity<String> response = algorithmRestTemplate.postForEntity(url, entity, String.class);

        if (!response.getStatusCode().is2xxSuccessful()) {
            throw new BusinessException("Python 服务返回非 2xx: " + response.getStatusCode() + " body=" + response.getBody());
        }

        String body = response.getBody();
        if (body == null || body.isBlank()) {
            throw new BusinessException("Python 服务返回空响应");
        }

        JSONObject json = JSONUtil.parseObj(body);

        // 统一解析 Python 服务响应格式: { "code": "00000", "data": {...}, "msg": "..." }
        String code = json.getStr("code");
        if (!"00000".equals(code)) {
            String msg = json.getStr("msg", "未知错误");
            log.error("Python 服务业务异常: code={}, msg={}", code, msg);
            throw new BusinessException("Python 服务错误 [" + code + "]: " + msg);
        }

        return json.getJSONObject("data");
    }

    // ==================== 熔断器 ====================

    private void checkCircuitBreaker(String path) {
        if (circuitState == CircuitState.OPEN) {
            long elapsed = System.currentTimeMillis() - lastFailureTime.get();
            if (elapsed > props.getCircuitBreakerHalfOpenDelay()) {
                log.info("熔断器进入半开状态，尝试探活");
                circuitState = CircuitState.HALF_OPEN;
            } else {
                throw new BusinessException("Python 算法服务熔断中，请稍后重试 (" +
                        (props.getCircuitBreakerHalfOpenDelay() - elapsed) / 1000 + "s 后恢复)");
            }
        }
    }

    private void recordSuccess(String path) {
        if (circuitState == CircuitState.HALF_OPEN) {
            log.info("半开探活成功，熔断器恢复");
            circuitState = CircuitState.CLOSED;
            totalCount.set(0);
            failureCount.set(0);
        }
    }

    private void recordFailure(String path) {
        int fails = failureCount.incrementAndGet();
        int total = totalCount.incrementAndGet();

        if (total >= props.getCircuitBreakerMinCalls()) {
            double failureRate = (double) fails / total * 100;
            if (failureRate >= props.getCircuitBreakerFailureRate()) {
                log.warn("熔断器触发: 失败率 {:.1f}% ({} / {}), 阈值 {}%",
                        failureRate, fails, total, props.getCircuitBreakerFailureRate());
                circuitState = CircuitState.OPEN;
                lastFailureTime.set(System.currentTimeMillis());
            }
        }
    }
}
