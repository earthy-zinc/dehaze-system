package com.pei.dehaze.common.interceptor;

import jakarta.annotation.PostConstruct;
import net.logstash.logback.argument.StructuredArguments;
import org.apache.ibatis.executor.statement.StatementHandler;
import org.apache.ibatis.mapping.BoundSql;
import org.apache.ibatis.plugin.Interceptor;
import org.apache.ibatis.plugin.Intercepts;
import org.apache.ibatis.plugin.Invocation;
import org.apache.ibatis.plugin.Signature;
import org.apache.ibatis.session.ResultHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.sql.Statement;
import java.util.Properties;

/**
 * SQL 审计日志拦截器。
 *
 * <p>拦截 StatementHandler 的 query/update/batch，记录结构化 SQL 执行日志：
 * 正常执行输出 INFO 级（message=SQL），超过阈值额外输出 WARN 级（message=SLOW_SQL）。
 * 请求上下文字段由 Logback 的 MDC 自动注入。SQL 以占位符形式记录，不含参数值，避免泄露敏感数据。
 */
@Intercepts({
        @Signature(type = StatementHandler.class, method = "query", args = {Statement.class, ResultHandler.class}),
        @Signature(type = StatementHandler.class, method = "update", args = {Statement.class}),
        @Signature(type = StatementHandler.class, method = "batch", args = {Statement.class})
})
@Component
public class SqlLogInterceptor implements Interceptor {

    private static final Logger log = LoggerFactory.getLogger("sql");

    @Value("${dehaze.sql-log.slow-threshold-ms:500}")
    private long slowThresholdMs;

    @Value("${dehaze.sql-log.level:INFO}")
    private String level;

    @PostConstruct
    public void init() {
        ch.qos.logback.classic.Logger sqlLogger =
                (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("sql");
        sqlLogger.setLevel(ch.qos.logback.classic.Level.toLevel(level, ch.qos.logback.classic.Level.INFO));
    }

    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        long start = System.currentTimeMillis();
        try {
            return invocation.proceed();
        } finally {
            long durationMs = System.currentTimeMillis() - start;
            StatementHandler handler = (StatementHandler) invocation.getTarget();
            BoundSql boundSql = handler.getBoundSql();
            String sql = boundSql.getSql();
            int rows = extractRows(invocation);
            if (durationMs >= slowThresholdMs) {
                log.warn("SLOW_SQL",
                        StructuredArguments.keyValue("sql", sql),
                        StructuredArguments.keyValue("duration_ms", durationMs),
                        StructuredArguments.keyValue("rows", rows),
                        StructuredArguments.keyValue("threshold_ms", slowThresholdMs));
            } else if (log.isInfoEnabled()) {
                log.info("SQL",
                        StructuredArguments.keyValue("sql", sql),
                        StructuredArguments.keyValue("duration_ms", durationMs),
                        StructuredArguments.keyValue("rows", rows));
            }
        }
    }

    private int extractRows(Invocation invocation) {
        Object[] args = invocation.getArgs();
        if (args.length > 0 && args[0] instanceof Statement statement) {
            try {
                return statement.getUpdateCount();
            } catch (Exception ignored) {
            }
        }
        return -1;
    }

    @Override
    public Object plugin(Object target) {
        return org.apache.ibatis.plugin.Plugin.wrap(target, this);
    }

    @Override
    public void setProperties(Properties properties) {
    }
}
