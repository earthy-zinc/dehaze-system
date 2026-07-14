package com.pei.dehaze.config;

import com.pei.dehaze.config.property.AlgorithmProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.hc.client5.http.config.ConnectionConfig;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManager;
import org.apache.hc.core5.util.TimeValue;
import org.apache.hc.core5.util.Timeout;
import org.slf4j.MDC;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.ClientHttpRequestInterceptor;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

/**
 * HTTP 客户端配置 —— 用于调用 Python 算法服务
 * <p>
 * 使用 Apache HttpClient 5 连接池，避免每次请求新建 TCP 连接。
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Slf4j
@Configuration
@RequiredArgsConstructor
public class RestClientConfig {

    private final AlgorithmProperties algorithmProperties;

    /**
     * 算法服务专用 RestTemplate（带连接池）
     */
    @Bean(name = "algorithmRestTemplate")
    public RestTemplate algorithmRestTemplate() {
        // 创建带连接池的 HTTP 连接管理器
        PoolingHttpClientConnectionManager connManager = new PoolingHttpClientConnectionManager();
        connManager.setMaxTotal(20);
        connManager.setDefaultMaxPerRoute(10);
        connManager.setDefaultConnectionConfig(ConnectionConfig.custom()
                .setConnectTimeout(Timeout.ofMilliseconds(algorithmProperties.getConnectTimeout()))
                .setSocketTimeout(Timeout.ofMilliseconds(algorithmProperties.getReadTimeout()))
                .build());

        // 创建带连接池的 HttpClient，定期清理空闲连接
        CloseableHttpClient httpClient = HttpClients.custom()
                .setConnectionManager(connManager)
                .evictIdleConnections(TimeValue.ofSeconds(30))
                .build();

        HttpComponentsClientHttpRequestFactory factory = new HttpComponentsClientHttpRequestFactory(httpClient);

        RestTemplate restTemplate = new RestTemplate(factory);
        restTemplate.getInterceptors().add(traceIdInterceptor());
        return restTemplate;
    }

    /**
     * 自动将 MDC 中的 traceId 注入到出站 HTTP 请求头，形成跨服务链路追踪
     */
    private ClientHttpRequestInterceptor traceIdInterceptor() {
        return (request, body, execution) -> {
            String traceId = MDC.get("traceId");
            if (traceId != null && !traceId.isBlank()) {
                request.getHeaders().set("X-Trace-Id", traceId);
            }
            return execution.execute(request, body);
        };
    }
}
