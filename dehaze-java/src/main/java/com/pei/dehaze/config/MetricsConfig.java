package com.pei.dehaze.config;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 业务指标配置
 * <p>
 * 注册预测/评估/Python调用/任务执行等核心业务 Prometheus 指标，
 * 对齐 Go 端 HTTP 中间件指标和 Python 端四大类指标。
 *
 * @author earthy-zinc
 * @since 2026-07-20
 */
@Configuration
public class MetricsConfig {

    /** 预测请求计数器（按状态区分成功/失败） */
    @Bean
    public Counter predictionSuccessCounter(MeterRegistry registry) {
        return Counter.builder("dehaze_prediction_total")
                .tag("status", "success")
                .description("预测请求成功次数")
                .register(registry);
    }

    @Bean
    public Counter predictionFailureCounter(MeterRegistry registry) {
        return Counter.builder("dehaze_prediction_total")
                .tag("status", "failure")
                .description("预测请求失败次数")
                .register(registry);
    }

    /** 评估请求计数器 */
    @Bean
    public Counter evaluationSuccessCounter(MeterRegistry registry) {
        return Counter.builder("dehaze_evaluation_total")
                .tag("status", "success")
                .description("评估请求成功次数")
                .register(registry);
    }

    @Bean
    public Counter evaluationFailureCounter(MeterRegistry registry) {
        return Counter.builder("dehaze_evaluation_total")
                .tag("status", "failure")
                .description("评估请求失败次数")
                .register(registry);
    }

    /** Python 算法服务调用计时器 */
    @Bean
    public Timer pythonCallTimer(MeterRegistry registry) {
        return Timer.builder("dehaze_python_call_duration")
                .description("Python 算法服务调用耗时")
                .register(registry);
    }

    /** 任务执行计数器（按状态） */
    @Bean
    public Counter taskCompletedCounter(MeterRegistry registry) {
        return Counter.builder("dehaze_task_total")
                .tag("status", "completed")
                .description("任务完成次数")
                .register(registry);
    }

    @Bean
    public Counter taskFailedCounter(MeterRegistry registry) {
        return Counter.builder("dehaze_task_total")
                .tag("status", "failed")
                .description("任务失败次数")
                .register(registry);
    }

    /** 文件上传计数器 */
    @Bean
    public Counter fileUploadCounter(MeterRegistry registry) {
        return Counter.builder("dehaze_file_upload_total")
                .description("文件上传次数")
                .register(registry);
    }
}
