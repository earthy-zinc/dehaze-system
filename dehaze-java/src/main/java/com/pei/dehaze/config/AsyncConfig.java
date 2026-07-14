package com.pei.dehaze.config;

import com.pei.dehaze.filter.TraceIdFilter;
import com.pei.dehaze.security.util.SystemSecurityContext;
import org.slf4j.MDC;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.task.TaskDecorator;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * 异步任务配置
 *
 * <p>TaskDecorator 同时传播 SecurityContext 和 MDC（traceId），
 * 保证 @Async 方法中日志链路追踪和权限上下文不中断。
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Configuration
@EnableAsync
public class AsyncConfig {

    /**
     * 数据集任务执行线程池
     */
    @Bean("datasetTaskExecutor")
    public Executor datasetTaskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(2);
        executor.setMaxPoolSize(4);
        executor.setQueueCapacity(10);
        executor.setThreadNamePrefix("dataset-async-");
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        executor.setTaskDecorator(asyncContextTaskDecorator());
        executor.initialize();
        return executor;
    }

    /**
     * 异步上下文传播装饰器
     *
     * <p>传播 SecurityContext（权限） + MDC traceId（日志链路）到异步线程。
     */
    @Bean
    public TaskDecorator asyncContextTaskDecorator() {
        return runnable -> {
            SecurityContext securityContext = SecurityContextHolder.getContext();
            Authentication authentication = securityContext.getAuthentication();
            String traceId = MDC.get(TraceIdFilter.MDC_TRACE_ID);
            return () -> {
                try {
                    if (authentication != null) {
                        SecurityContextHolder.getContext().setAuthentication(authentication);
                    } else {
                        SystemSecurityContext.setSystemContext();
                    }
                    if (traceId != null) {
                        MDC.put(TraceIdFilter.MDC_TRACE_ID, traceId);
                    }
                    runnable.run();
                } finally {
                    SecurityContextHolder.clearContext();
                    MDC.remove(TraceIdFilter.MDC_TRACE_ID);
                }
            };
        };
    }
}
