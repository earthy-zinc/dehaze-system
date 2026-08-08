package com.pei.dehaze.service.impl;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.form.ClientLogBatchForm;
import com.pei.dehaze.model.form.ClientLogEntryForm;
import com.pei.dehaze.security.util.SecurityUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mockStatic;

/**
 * ClientLogServiceImpl 单元测试
 * <p>
 * 验证前端日志接收的写入逻辑：匿名过滤、字段截断、级别映射、
 * buildFields 字段过滤、MDC 上下文覆盖与还原。
 */
@DisplayName("ClientLogServiceImpl 单元测试")
@ExtendWith(MockitoExtension.class)
class ClientLogServiceImplTest {

    private ClientLogServiceImpl service;
    private ListAppender<ILoggingEvent> appender;

    @BeforeEach
    void setUp() {
        service = new ClientLogServiceImpl();
        Logger clientLogger = (Logger) LoggerFactory.getLogger("client-log");
        clientLogger.setLevel(Level.INFO);
        appender = new ListAppender<>();
        appender.start();
        clientLogger.addAppender(appender);
    }

    @AfterEach
    void tearDown() {
        Logger clientLogger = (Logger) LoggerFactory.getLogger("client-log");
        clientLogger.detachAppender(appender);
    }

    private ClientLogEntryForm entry(String level, String traceId) {
        ClientLogEntryForm e = new ClientLogEntryForm();
        e.setLevel(level);
        e.setTraceId(traceId);
        e.setMessage("test message");
        return e;
    }

    @Test
    @DisplayName("collect - 空日志列表抛出业务异常")
    void collect_emptyThrows() {
        assertThatThrownBy(() -> service.collect(new ClientLogBatchForm()))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("日志列表不能为空");
    }

    @Test
    @DisplayName("collect - 匿名 ERROR 且带 trace_id 正常落盘")
    void collect_anonymousErrorWithTraceIdWrites() {
        ClientLogBatchForm form = new ClientLogBatchForm();
        form.setLogs(List.of(entry("ERROR", "trace-abc")));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(null);
            service.collect(form);
        }

        assertThat(appender.list).hasSize(1);
        ILoggingEvent event = appender.list.get(0);
        assertThat(event.getLevel()).isEqualTo(Level.ERROR);
        assertThat(event.getMessage()).isEqualTo("test message");
    }

    @Test
    @DisplayName("collect - 匿名 WARN（非 ERROR）被丢弃")
    void collect_anonymousWarnDropped() {
        ClientLogBatchForm form = new ClientLogBatchForm();
        form.setLogs(List.of(entry("WARN", "trace-abc")));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(null);
            service.collect(form);
        }

        assertThat(appender.list).isEmpty();
    }

    @Test
    @DisplayName("collect - 匿名 ERROR 但无 trace_id 被丢弃")
    void collect_anonymousErrorWithoutTraceIdDropped() {
        ClientLogBatchForm form = new ClientLogBatchForm();
        form.setLogs(List.of(entry("ERROR", null)));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(null);
            service.collect(form);
        }

        assertThat(appender.list).isEmpty();
    }

    @Test
    @DisplayName("collect - 已登录用户 INFO 也正常落盘（不受匿名限制）")
    void collect_loggedInInfoWrites() {
        ClientLogBatchForm form = new ClientLogBatchForm();
        form.setLogs(List.of(entry("INFO", null)));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(42L);
            service.collect(form);
        }

        assertThat(appender.list).hasSize(1);
        assertThat(appender.list.get(0).getLevel()).isEqualTo(Level.INFO);
    }

    @Test
    @DisplayName("collect - 超长 message 与 error_stack 同时被截断")
    void collect_longFieldsTruncated() {
        ClientLogEntryForm e = entry("ERROR", "trace-abc");
        e.setMessage("m".repeat(3000));
        e.setErrorStack("s".repeat(10000));

        ClientLogBatchForm form = new ClientLogBatchForm();
        form.setLogs(List.of(e));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(null);
            service.collect(form);
        }

        assertThat(appender.list).hasSize(1);
        ILoggingEvent event = appender.list.get(0);
        assertThat(event.getMessage()).hasSize(2000);
        // buildFields 返回的 error_stack 应被截断到 8000
        Map<String, Object> fields = service.buildFields(e, null, "trace-abc");
        assertThat(((String) fields.get("error_stack")).length()).isEqualTo(8000);
    }

    @Test
    @DisplayName("collect - 匿名多级别混合仅 ERROR 落盘")
    void collect_mixedAnonymousOnlyErrorWrites() {
        ClientLogBatchForm form = new ClientLogBatchForm();
        form.setLogs(List.of(
                entry("ERROR", "trace-1"),
                entry("INFO", "trace-2"),
                entry("WARN", "trace-3")
        ));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(null);
            service.collect(form);
        }

        assertThat(appender.list).hasSize(1);
        assertThat(appender.list.get(0).getLevel()).isEqualTo(Level.ERROR);
    }

    @Test
    @DisplayName("logAtLevel - 级别大小写不敏感，空白默认 INFO")
    void collect_levelNormalization() {
        ClientLogBatchForm form = new ClientLogBatchForm();
        form.setLogs(List.of(
                entry("error", "t1"),  // 小写应识别为 ERROR
                entry("Warn", "t2"),   // 混合大小写应识别为 WARN
                entry("", "t3"),       // 空白应默认 INFO
                entry(null, "t4")      // null 应默认 INFO
        ));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(1L);
            service.collect(form);
        }

        assertThat(appender.list).hasSize(4);
        assertThat(appender.list.get(0).getLevel()).isEqualTo(Level.ERROR);
        assertThat(appender.list.get(1).getLevel()).isEqualTo(Level.WARN);
        assertThat(appender.list.get(2).getLevel()).isEqualTo(Level.INFO);
        assertThat(appender.list.get(3).getLevel()).isEqualTo(Level.INFO);
    }

    @Test
    @DisplayName("buildFields - 空白/null 字符串字段不进入 fields")
    void collect_blankStringFieldsExcluded() {
        ClientLogEntryForm e = entry("ERROR", "trace-abc");
        e.setApp("react");
        e.setUrl("   ");   // 空白：不应进入
        e.setUserAgent(null);  // null：不应进入
        e.setMethod("POST");
        e.setPath(null);   // null：不应进入

        Map<String, Object> fields = service.buildFields(e, null, "trace-abc");

        assertThat(fields).containsEntry("app", "react");
        assertThat(fields).containsEntry("method", "POST");
        assertThat(fields).doesNotContainKey("url");
        assertThat(fields).doesNotContainKey("user_agent");
        assertThat(fields).doesNotContainKey("path");
    }

    @Test
    @DisplayName("buildFields - 数值字段 null 不进入，有值进入")
    void collect_numericFieldsNullable() {
        ClientLogEntryForm e = entry("ERROR", "trace-abc");
        e.setStatus(500);
        e.setDuration(1203.5);
        e.setMetricValue(null);  // null：不应进入

        Map<String, Object> fields = service.buildFields(e, null, "trace-abc");

        assertThat(fields).containsEntry("status", 500);
        assertThat(fields).containsEntry("duration", 1203.5);
        assertThat(fields).doesNotContainKey("metric_value");
    }

    @Test
    @DisplayName("MDC - 写入日志时覆盖 trace_id/user_id，写入后还原原值")
    void collect_mdcOverwriteAndRestore() {
        // 预设请求上下文的 MDC
        MDC.put("trace_id", "request-trace");
        MDC.put("user_id", "999");

        try {
            ClientLogEntryForm e = entry("ERROR", "log-trace");
            ClientLogBatchForm form = new ClientLogBatchForm();
            form.setLogs(List.of(e));

            try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
                mocked.when(SecurityUtils::getUserId).thenReturn(42L);
                service.collect(form);
            }

            // 落盘时应使用日志自身的 trace_id / user_id（被覆盖）
            assertThat(appender.list).hasSize(1);
            // 写入完成后 MDC 应还原到请求上下文原值
            assertThat(MDC.get("trace_id")).isEqualTo("request-trace");
            assertThat(MDC.get("user_id")).isEqualTo("999");
        } finally {
            MDC.clear();
        }
    }

    @Test
    @DisplayName("MDC - 写入后还原时若原值不存在则移除（不残留）")
    void collect_mdcRemoveIfOriginAbsent() {
        // 请求上下文无 MDC（首次调用）
        MDC.clear();

        ClientLogEntryForm e = entry("ERROR", "log-trace");
        ClientLogBatchForm form = new ClientLogBatchForm();
        form.setLogs(List.of(e));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(null);
            service.collect(form);
        }

        // 写入完成后 MDC 不应残留（原值都是 null）
        assertThat(MDC.get("trace_id")).isNull();
        assertThat(MDC.get("user_id")).isNull();
    }
}
