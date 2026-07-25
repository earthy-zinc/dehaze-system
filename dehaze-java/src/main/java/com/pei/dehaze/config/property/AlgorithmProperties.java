package com.pei.dehaze.config.property;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * Python 算法服务配置属性
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Data
@Component
@ConfigurationProperties(prefix = "algorithm.python")
public class AlgorithmProperties {

    /** Python 算法服务基础 URL（如 http://127.0.0.1:8991） */
    private String baseUrl = "http://127.0.0.1:8991";

    /** 连接超时（毫秒） */
    private int connectTimeout = 5000;

    /** 读取超时（毫秒），预测/评估可较长时间 */
    private int readTimeout = 30000;

    /** 最大重试次数 */
    private int maxRetry = 3;

    /** 重试初始退避时间（毫秒） */
    private long retryBackoff = 1000;

    /** 熔断器失败率阈值（百分比，0-100） */
    private int circuitBreakerFailureRate = 50;

    /** 熔断器最小调用次数后才开始统计 */
    private int circuitBreakerMinCalls = 10;

    /** 熔断后半开状态等待时间（毫秒） */
    private long circuitBreakerHalfOpenDelay = 30000;

    /** 预测端点路径 */
    private String predictPath = "/api/v1/prediction";

    /** 评估端点路径 */
    private String evaluatePath = "/api/v1/evaluation";

    /** 服务间调用 API Key（M2M 认证） */
    private String apiKey;
}
