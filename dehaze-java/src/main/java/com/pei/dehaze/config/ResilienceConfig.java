package com.pei.dehaze.config;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.config.property.AlgorithmProperties;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;

/**
 * 弹性容错配置（Resilience4j）
 *
 * <p>使用 Resilience4j 替换自研熔断器，提供滑动窗口统计、半开探活等标准能力。
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Configuration
@RequiredArgsConstructor
public class ResilienceConfig {

    private final AlgorithmProperties algorithmProperties;

    /**
     * Python 算法服务熔断器
     *
     * <p>配置参数映射自 {@link AlgorithmProperties}：
     * <ul>
     *   <li>{@code circuitBreakerFailureRate} → failureRateThreshold（百分比）</li>
     *   <li>{@code circuitBreakerMinCalls} → minimumNumberOfCalls</li>
     *   <li>{@code circuitBreakerHalfOpenDelay} → waitDurationInOpenState</li>
     * </ul>
     *
     * <p>仅对网络层异常（连接超时/拒绝/读取超时）计数，忽略 Python 业务层错误
     * （如图片下载 404、参数校验失败等），避免下游数据问题误触发熔断。
     */
    @Bean
    public CircuitBreaker pythonAlgorithmCircuitBreaker() {
        CircuitBreakerConfig config = CircuitBreakerConfig.custom()
                .failureRateThreshold(algorithmProperties.getCircuitBreakerFailureRate())
                .minimumNumberOfCalls(algorithmProperties.getCircuitBreakerMinCalls())
                .waitDurationInOpenState(Duration.ofMillis(algorithmProperties.getCircuitBreakerHalfOpenDelay()))
                .slidingWindowSize(algorithmProperties.getCircuitBreakerMinCalls())
                .permittedNumberOfCallsInHalfOpenState(5)
                .ignoreExceptions(BusinessException.class)
                .build();
        return CircuitBreaker.of("pythonAlgorithm", config);
    }
}
