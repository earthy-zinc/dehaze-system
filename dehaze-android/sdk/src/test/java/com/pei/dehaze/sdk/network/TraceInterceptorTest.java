package com.pei.dehaze.sdk.network;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.pei.dehaze.sdk.logger.LogEntry;
import com.pei.dehaze.sdk.logger.LogLevel;
import com.pei.dehaze.sdk.logger.LogTransport;
import com.pei.dehaze.sdk.logger.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import okhttp3.Interceptor;
import okhttp3.Protocol;
import okhttp3.Request;
import okhttp3.Response;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * API 失败日志级别：401 是鉴权结果（未登录 / 会话失效），按 INFO 记录（不上报远端）；
 * 其余失败仍按 ERROR 上报，避免把预期结果混进真实故障。
 */
public class TraceInterceptorTest {

    private CaptureTransport transport;

    @Before
    public void setUp() {
        transport = new CaptureTransport();
        Logger.init("android", "1.0.0", Collections.singletonList(transport));
    }

    @After
    public void tearDown() {
        Logger.resetForTest();
    }

    @Test
    public void unauthenticatedResponseLoggedAsInfo() throws IOException {
        intercept(401, "/api/v1/auth/me");

        assertEquals(1, transport.logs.size());
        LogEntry entry = transport.logs.get(0);
        assertEquals(LogLevel.INFO, entry.getLevel());
        assertEquals("API_UNAUTHORIZED", entry.getMessage());
        assertEquals("GET", entry.getMethod());
        assertEquals("/api/v1/auth/me", entry.getPath());
        assertEquals(Integer.valueOf(401), entry.getStatus());
    }

    @Test
    public void otherFailuresLoggedAsError() throws IOException {
        intercept(500, "/api/v1/ai/providers");

        assertEquals(1, transport.logs.size());
        LogEntry entry = transport.logs.get(0);
        assertEquals(LogLevel.ERROR, entry.getLevel());
        assertEquals("API_ERROR", entry.getMessage());
        assertEquals("api_interceptor", entry.getErrorSource());
    }

    private void intercept(int status, String path) throws IOException {
        Request request = new Request.Builder().url("https://dehaze.example.com" + path).build();
        Interceptor.Chain chain = mock(Interceptor.Chain.class);
        when(chain.request()).thenReturn(request);
        when(chain.proceed(any(Request.class)))
                .thenReturn(new Response.Builder()
                        .request(request)
                        .protocol(Protocol.HTTP_1_1)
                        .code(status)
                        .message("status " + status)
                        .build());

        new TraceInterceptor().intercept(chain);
    }

    /** 仅捕获 log 输出、不批量上报的 transport（观测采样前的本地输出） */
    private static class CaptureTransport implements LogTransport {
        final List<LogEntry> logs = new ArrayList<>();

        @Override
        public void log(LogEntry entry) {
            logs.add(entry);
        }
    }
}
