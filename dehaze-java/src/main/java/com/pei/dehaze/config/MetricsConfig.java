package com.pei.dehaze.config;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 业务指标配置
 * <p>
 * 预测/评估/任务/文件上传计数器由各业务 Service 通过 MeterRegistry 直接创建并递增：
 * <ul>
 *   <li>{@code dehaze_prediction_total{status}} — PredLogAsyncTask / SysPredLogServiceImpl</li>
 *   <li>{@code dehaze_evaluation_total{status}} — EvalLogAsyncTask</li>
 *   <li>{@code dehaze_task_total{status}} — TaskServiceImpl</li>
 *   <li>{@code dehaze_file_upload_total} — SysFileServiceImpl</li>
 * </ul>
 *
 * @author earthy-zinc
 * @since 2026-07-20
 */
@Configuration
public class MetricsConfig {

    /** Python 算法服务调用计时器 */
    @Bean
    public Timer pythonCallTimer(MeterRegistry registry) {
        return Timer.builder("dehaze_python_call_duration")
                .description("Python 算法服务调用耗时")
                .register(registry);
    }
}
