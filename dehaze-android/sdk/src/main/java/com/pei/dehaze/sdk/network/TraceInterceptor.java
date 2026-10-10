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
 * - 失败请求交 Logger 上报（关联 method/path/status/duration/code）；401 属鉴权结果，按 INFO 记录不上报
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

        // 401 是鉴权结果（未登录/会话失效），不是服务端故障：按 INFO 记录（本地可见、不上报远端），
        // 计成 ERROR 会把预期结果混进真实故障
        boolean unauthenticated = status != null && status == 401;
        LogEntry entry = new LogEntry(
                        unauthenticated ? LogLevel.INFO : LogLevel.ERROR,
                        unauthenticated ? "API_UNAUTHORIZED" : "API_ERROR", "", "")
                .setTraceId(TraceManager.getCurrentTraceId())
                .setMethod(request.method().toUpperCase())
                .setPath(path)
                .setStatus(status)
                .setDuration((double) durationMs);
        if (unauthenticated) {
            Logger.getInstance().info(entry.getMessage(), entry);
            return;
        }
        entry.setErrorType("api")
                .setErrorSource("api_interceptor")
                .setErrorStack(error != null ? error.toString() : null);
        Logger.getInstance().error(entry.getMessage(), entry);
    }
}
