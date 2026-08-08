package com.pei.dehaze.sdk.network;

import com.pei.dehaze.sdk.logger.LogEntry;
import com.pei.dehaze.sdk.logger.LogLevel;
import com.pei.dehaze.sdk.logger.Logger;
import com.pei.dehaze.sdk.logger.TraceManager;

import java.io.IOException;

import okhttp3.Interceptor;
import okhttp3.Request;
import okhttp3.Response;

/**
 * OkHttp 拦截器：
 * - 请求拦截：生成/复用 trace_id 注入 X-Trace-Id 头
 * - 响应拦截：读取响应头 X-Trace-Id 与本地对齐
 * - 失败请求交 Logger 上报（关联 method/path/status/duration/code）
 */
public class TraceInterceptor implements Interceptor {

    @Override
    public Response intercept(Chain chain) throws IOException {
        Request original = chain.request();

        // 生成/复用 trace_id 注入请求头
        String traceId = TraceManager.ensureTraceId();
        Request.Builder requestBuilder = original.newBuilder()
                .header("X-Trace-Id", traceId);

        long startTime = System.currentTimeMillis();
        try {
            Response response = chain.proceed(requestBuilder.build());

            // 读取响应头 X-Trace-Id 对齐
            String responseTraceId = response.header("X-Trace-Id");
            if (responseTraceId != null && !responseTraceId.isEmpty()) {
                TraceManager.alignTraceId(responseTraceId);
            }

            // 非成功响应也交 Logger（HTTP 层失败）
            if (!response.isSuccessful() && Logger.isInitialized()) {
                logApiFailure(original, response.code(), null,
                        System.currentTimeMillis() - startTime);
            }
            return response;
        } catch (IOException e) {
            // 网络异常
            if (Logger.isInitialized()) {
                logApiFailure(original, null, e,
                        System.currentTimeMillis() - startTime);
            }
            throw e;
        }
    }

    private void logApiFailure(Request request, Integer status, IOException error,
                               long durationMs) {
        String path = request.url().encodedPath();
        // 避免为日志上报 API 自身再记录（防循环）
        if (path.contains("/api/v1/logs/client")) {
            return;
        }

        LogEntry entry = new LogEntry(LogLevel.ERROR, "API_ERROR", "", "")
                .setTraceId(TraceManager.getCurrentTraceId())
                .setMethod(request.method().toUpperCase())
                .setPath(path)
                .setStatus(status)
                .setDuration((double) durationMs)
                .setErrorType("api")
                .setErrorSource("api_interceptor")
                .setErrorStack(error != null ? error.toString() : null);
        Logger.getInstance().error("API_ERROR", entry);
    }
}
