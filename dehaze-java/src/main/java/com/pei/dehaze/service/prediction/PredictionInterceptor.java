package com.pei.dehaze.service.prediction;

import java.util.Optional;

/**
 * 预测主流程拦截器，用于在调用算法前插入可插拔的快速返回逻辑。
 * <p>
 * 命中（返回非 empty）则短路主流程，不调用 Python 算法服务；
 * 未命中（返回 empty）则继续走主流程。
 */
public interface PredictionInterceptor {

    Optional<InterceptedResult> intercept(PredictionContext context);
}
