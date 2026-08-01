package com.pei.dehaze.service.prediction;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Optional;

/**
 * 预测拦截器责任链：按 Spring 注入顺序执行，第一个命中即短路。
 * <p>
 * 通过构造器注入 {@code List<PredictionInterceptor>}，
 * 新增插件只需实现接口并标注 {@code @Component} 即可自动注册。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class PredictionInterceptorChain {

    private final List<PredictionInterceptor> interceptors;

    public Optional<InterceptedResult> intercept(PredictionContext context) {
        for (PredictionInterceptor interceptor : interceptors) {
            try {
                Optional<InterceptedResult> result = interceptor.intercept(context);
                if (result.isPresent()) {
                    log.debug("预测拦截器命中: {} -> resultUrl={}",
                            interceptor.getClass().getSimpleName(),
                            result.get().getResultUrl());
                    return result;
                }
            } catch (Exception e) {
                log.warn("预测拦截器执行异常，跳过: {}", interceptor.getClass().getSimpleName(), e);
            }
        }
        return Optional.empty();
    }
}
