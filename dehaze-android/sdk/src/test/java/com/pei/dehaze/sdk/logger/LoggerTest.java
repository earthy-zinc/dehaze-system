package com.pei.dehaze.sdk.logger;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

/**
 * ERROR 去重 + 次数汇总测试。
 *
 * 去重判定为同步逻辑，可立即断言；汇总补发依赖 ScheduledExecutorService 真实定时器，
 * 用 CountDownLatch 等待 10s 窗口触发后断言。
 */
public class LoggerTest {

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
    public void 相同fingerprint在10s窗口内只输出首条_重复被去重() {
        Logger logger = Logger.getInstance();
        logger.error("RenderFlex overflowed", extras("stack-A", "dart", null));
        logger.error("RenderFlex overflowed", extras("stack-A", "dart", null));
        logger.error("RenderFlex overflowed", extras("stack-A", "dart", null));

        List<LogEntry> real = nonSummaryLogs();
        assertEquals(1, real.size());
        assertEquals("RenderFlex overflowed", real.get(0).getMessage());
    }

    @Test
    public void 窗口结束时补发汇总条目_dedupCount标记总次数_message标注重复次数() throws InterruptedException {
        Logger logger = Logger.getInstance();
        logger.error("RenderFlex overflowed", extras("stack-A", "dart", "flutter_error"));
        for (int i = 0; i < 5; i++) {
            logger.error("RenderFlex overflowed", extras("stack-A", "dart", null));
        }

        // 等待 10s 窗口定时器触发补发（留 2s 余量）
        Thread.sleep(12_000);

        List<LogEntry> summaries = summaryLogs();
        assertEquals(1, summaries.size());
        assertEquals(Integer.valueOf(6), summaries.get(0).getDedupCount());
        assertEquals("RenderFlex overflowed (10s 内重复 5 次)", summaries.get(0).getMessage());
        assertEquals("stack-A", summaries.get(0).getErrorStack());
        assertEquals("dart", summaries.get(0).getErrorType());
        assertEquals("flutter_error", summaries.get(0).getErrorSource());
    }

    @Test
    public void 单次命中无重复时不补发汇总_避免噪声() throws InterruptedException {
        Logger logger = Logger.getInstance();
        logger.error("one-shot-error", null);
        Thread.sleep(12_000);

        assertEquals(1, transport.logs.size());
        assertTrue(summaryLogs().isEmpty());
    }

    @Test
    public void 不同fingerprint不去重_各自独立输出() {
        Logger logger = Logger.getInstance();
        logger.error("error-A", extras("stack-A", null, null));
        logger.error("error-B", extras("stack-B", null, null));

        List<LogEntry> real = nonSummaryLogs();
        assertEquals(2, real.size());
    }

    @Test
    public void 不同fingerprint到来时先补发上一轮汇总() {
        Logger logger = Logger.getInstance();
        logger.error("error-A", extras("stack-A", null, null));
        logger.error("error-A", extras("stack-A", null, null));
        logger.error("error-A", extras("stack-A", null, null));
        // 不同 fingerprint 到来：先补发 A 的汇总，再输出 B
        logger.error("error-B", extras("stack-B", null, null));

        List<LogEntry> summaries = summaryLogs();
        assertEquals(1, summaries.size());
        assertEquals(Integer.valueOf(3), summaries.get(0).getDedupCount());
        assertTrue(summaries.get(0).getMessage().contains("error-A"));

        // A 首条 + B 首条
        List<LogEntry> real = nonSummaryLogs();
        assertEquals(2, real.size());
    }

    @Test
    public void WARN_INFO不参与去重() {
        Logger logger = Logger.getInstance();
        logger.warn("same-warn", null);
        logger.warn("same-warn", null);
        logger.info("same-info", null);
        logger.info("same-info", null);

        // WARN/INFO 不去重，全部经 emit 输出到 transport
        assertEquals(4, transport.logs.size());
    }

    @Test
    public void 汇总条目经emit输出_toJson携带dedup_count字段() throws InterruptedException {
        Logger logger = Logger.getInstance();
        logger.error("storm-error", extras("stack-A", null, null));
        for (int i = 0; i < 9; i++) {
            logger.error("storm-error", extras("stack-A", null, null));
        }
        Thread.sleep(12_000);

        List<LogEntry> summaries = summaryLogs();
        assertEquals(1, summaries.size());
        assertEquals(Integer.valueOf(10), summaries.get(0).getDedupCount());
        assertTrue(summaries.get(0).toJson().has("dedup_count"));
        assertEquals(10, summaries.get(0).toJson().get("dedup_count").getAsInt());
    }

    @Test
    public void 窗口过期后相同fingerprint视为新burst_先补发上一轮汇总再输出新首条() throws InterruptedException {
        Logger logger = Logger.getInstance();
        logger.error("recurring-error", extras("stack-A", null, null));
        logger.error("recurring-error", extras("stack-A", null, null)); // 重复 1 次

        // 等待窗口过期
        Thread.sleep(12_000);
        // 再触发相同错误 → 视为新 burst
        logger.error("recurring-error", extras("stack-A", null, null));

        List<LogEntry> summaries = summaryLogs();
        assertEquals(1, summaries.size());
        assertEquals(Integer.valueOf(2), summaries.get(0).getDedupCount());

        // 第一轮首条 + 第二轮首条
        List<LogEntry> real = nonSummaryLogs();
        assertEquals(2, real.size());
    }

    private LogEntry extras(String errorStack, String errorType, String errorSource) {
        return new LogEntry(LogLevel.ERROR, "", "", "")
                .setErrorStack(errorStack)
                .setErrorType(errorType)
                .setErrorSource(errorSource);
    }

    private List<LogEntry> nonSummaryLogs() {
        List<LogEntry> result = new ArrayList<>();
        for (LogEntry l : transport.logs) {
            if (l.getDedupCount() == null) result.add(l);
        }
        return result;
    }

    private List<LogEntry> summaryLogs() {
        List<LogEntry> result = new ArrayList<>();
        for (LogEntry l : transport.logs) {
            if (l.getDedupCount() != null) result.add(l);
        }
        return result;
    }

    /** 仅捕获 log 输出、不批量上报的 transport（用于观测采样前的本地输出） */
    static class CaptureTransport implements LogTransport {
        final List<LogEntry> logs = new ArrayList<>();

        @Override
        public void log(LogEntry entry) {
            logs.add(entry);
        }
    }
}
