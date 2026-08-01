package com.pei.dehaze.config;

import com.pei.dehaze.filter.TraceIdFilter;
import com.pei.dehaze.security.util.SystemSecurityContext;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.aop.interceptor.AsyncUncaughtExceptionHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.task.TaskDecorator;
import org.springframework.scheduling.annotation.AsyncConfigurer;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.lang.reflect.Method;
import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * 异步任务配置
 *
 * <p>TaskDecorator 同时传播 SecurityContext 和 MDC（traceId），
 * 保证 @Async 方法中日志链路追踪和权限上下文不中断。
 *
 * <p>线程池参数说明：
 * <ul>
 *   <li>core=4, max=8：支持多个导出任务并发执行</li>
 *   <li>queue=50：缓冲突发请求，避免过早触发拒绝策略</li>
 *   <li>CallerRunsPolicy：队列满时由调用线程执行（背压），不会丢弃任务</li>
 * </ul>
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Slf4j
@Configuration
@EnableAsync
public class AsyncConfig implements AsyncConfigurer {

    /**
     * 数据集任务执行线程池
     */
    @Bean("datasetTaskExecutor")
    public Executor datasetTaskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(4);
        executor.setMaxPoolSize(8);
        executor.setQueueCapacity(50);
        executor.setKeepAliveSeconds(60);
        executor.setThreadNamePrefix("dataset-async-");
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        executor.setTaskDecorator(asyncContextTaskDecorator());
        executor.initialize();
        return executor;
    }

    /**
     * 消息推送执行线程池（APP推送/邮件/短信异步执行，不阻塞主流程）
     */
    @Bean("pushTaskExecutor")
    public Executor pushTaskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(2);
        executor.setMaxPoolSize(4);
        executor.setQueueCapacity(200);
        executor.setKeepAliveSeconds(60);
        executor.setThreadNamePrefix("message-push-");
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        executor.setTaskDecorator(asyncContextTaskDecorator());
        executor.initialize();
        return executor;
    }

    /**
     * 异步未捕获异常处理器：记录异常日志，避免静默丢失
     */
    @Override
    public AsyncUncaughtExceptionHandler getAsyncUncaughtExceptionHandler() {
        return (Throwable ex, Method method, Object... params) ->
                log.error("异步任务未捕获异常: method={}, params={}", method.getName(), params, ex);
    }

    /**
     * 异步上下文传播装饰器
     *
     * <p>传播 SecurityContext（权限） + MDC trace_id/method/path/ip/user_agent（日志链路）+ user_id（认证层）到异步线程。
     */
    @Bean
    public TaskDecorator asyncContextTaskDecorator() {
        return runnable -> {
            SecurityContext securityContext = SecurityContextHolder.getContext();
            Authentication authentication = securityContext.getAuthentication();
            String traceId = MDC.get(TraceIdFilter.MDC_TRACE_ID);
            String method = MDC.get(TraceIdFilter.MDC_METHOD);
            String path = MDC.get(TraceIdFilter.MDC_PATH);
            String ip = MDC.get(TraceIdFilter.MDC_IP);
            String userAgent = MDC.get(TraceIdFilter.MDC_USER_AGENT);
            String userId = MDC.get("user_id");
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
                    if (method != null) {
                        MDC.put(TraceIdFilter.MDC_METHOD, method);
                    }
                    if (path != null) {
                        MDC.put(TraceIdFilter.MDC_PATH, path);
                    }
                    if (ip != null) {
                        MDC.put(TraceIdFilter.MDC_IP, ip);
                    }
                    if (userAgent != null) {
                        MDC.put(TraceIdFilter.MDC_USER_AGENT, userAgent);
                    }
                    if (userId != null) {
                        MDC.put("user_id", userId);
                    }
                    runnable.run();
                } finally {
                    SecurityContextHolder.clearContext();
                    MDC.remove(TraceIdFilter.MDC_TRACE_ID);
                    MDC.remove(TraceIdFilter.MDC_METHOD);
                    MDC.remove(TraceIdFilter.MDC_PATH);
                    MDC.remove(TraceIdFilter.MDC_IP);
                    MDC.remove(TraceIdFilter.MDC_USER_AGENT);
                    MDC.remove("user_id");
                }
            };
        };
    }
}
