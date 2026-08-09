package com.pei.dehaze.sdk.logger;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/**
 * Android 端日志 Logger 单例（与 Flutter 端行为对齐，§3.6）。
 *
 * 多 transport：
 * - ConsoleTransport（始终开启，Logcat）
 * - FileTransport（开发 7 天 / 生产 3 天兜底）
 * - RemoteTransport（生产批量上报）
 *
 * 采样限流（§3.4）：ERROR 100% / WARN 50% / INFO 不上报；60s 内最多 20 条。
 * 队列上限 500 条，满 10 条立即上报，30s 定时器，失败指数退避。
 */
public class Logger {

    private static volatile Logger instance;

    private final String app;
    private final String appVersion;
    private final List<LogTransport> transports;
    private final List<LogEntry> queue = new ArrayList<>();
    private final List<Long> sentTimestamps = new ArrayList<>();

    private static final int MAX_QUEUE = 500;
    private static final int FLUSH_THRESHOLD = 10;
    private static final long FLUSH_INTERVAL_SECONDS = 30;
    private static final long RATE_LIMIT_WINDOW_MS = 60_000;
    private static final int RATE_LIMIT_MAX = 20;
    private static final int MAX_MESSAGE_LENGTH = 2000;
    private static final int MAX_STACK_LENGTH = 8000;
    private static final int INITIAL_BACKOFF_MS = 1000;
    private static final int MAX_BACKOFF_MS = 60_000;
    private static final long DEDUP_WINDOW_MS = 10_000L;

    // 采样率（%）：ERROR 100% / WARN 50% / INFO 不上报
    private static int sampleRate(LogLevel level) {
        switch (level) {
            case ERROR: return 100;
            case WARN: return 50;
            default: return 0;
        }
    }

    private boolean flushing = false;
    private int backoffMs = INITIAL_BACKOFF_MS;
    private final ScheduledExecutorService executor =
            Executors.newSingleThreadScheduledExecutor();

    // ERROR 去重：相同 message + error_stack fingerprint 在 10s 窗口内只输出首条，
    // 窗口结束时若存在重复则补发一条汇总（dedupCount 标记总次数），避免日志风暴同时保留次数信息
    private int lastErrorFingerprint = 0;
    private long lastErrorTime = 0L;
    private int errorDedupCount = 0;
    private LogEntry lastDedupEntry = null;
    private ScheduledFuture<?> dedupSummaryFuture = null;

    private Logger(String app, String appVersion, List<LogTransport> transports) {
        this.app = app;
        this.appVersion = appVersion;
        this.transports = transports;
        this.executor.scheduleWithFixedDelay(this::flush, FLUSH_INTERVAL_SECONDS,
                FLUSH_INTERVAL_SECONDS, TimeUnit.SECONDS);
    }

    /**
     * 初始化 Logger 单例（在 Application.onCreate 中调用）。
     *
     * @param app          前端项目标识（固定 "android"）
     * @param appVersion   应用版本号
     * @param transports   transport 列表（由调用方按环境组装）
     */
    public static synchronized void init(String app, String appVersion,
                                         List<LogTransport> transports) {
        if (instance == null) {
            instance = new Logger(app, appVersion, transports);
        } else {
            // 重新配置
            instance.transports.clear();
            instance.transports.addAll(transports);
        }
    }

    /**
     * 重置单例与所有内部状态（去重计数、定时任务、队列）。
     * 仅供测试隔离使用，生产环境不应调用。
     */
    public static void resetForTest() {
        if (instance == null) return;
        if (instance.dedupSummaryFuture != null) {
            instance.dedupSummaryFuture.cancel(false);
            instance.dedupSummaryFuture = null;
        }
        instance.executor.shutdownNow();
        instance.errorDedupCount = 0;
        instance.lastErrorFingerprint = 0;
        instance.lastErrorTime = 0L;
        instance.lastDedupEntry = null;
        synchronized (instance.queue) {
            instance.queue.clear();
        }
        instance.sentTimestamps.clear();
        instance = null;
    }

    public static Logger getInstance() {
        if (instance == null) {
            throw new IllegalStateException("Logger not initialized. Call Logger.init() first.");
        }
        return instance;
    }

    public static boolean isInitialized() {
        return instance != null;
    }

    public List<LogTransport> getTransports() {
        return transports;
    }

    // ==================== log 入口 ====================

    public void log(LogLevel level, String message, LogEntry extras) {
        String msg = truncate(message, MAX_MESSAGE_LENGTH);
        LogEntry entry = new LogEntry(level, msg, app, appVersion);
        if (extras != null) {
            entry.setUrl(extras.getUrl())
                    .setUserAgent(extras.getUserAgent())
                    .setTraceId(extras.getTraceId())
                    .setUserId(extras.getUserId())
                    .setErrorType(extras.getErrorType())
                    .setErrorSource(extras.getErrorSource())
                    .setErrorStack(extras.getErrorStack() != null
                            ? truncate(extras.getErrorStack(), MAX_STACK_LENGTH)
                            : null)
                    .setMethod(extras.getMethod())
                    .setPath(extras.getPath())
                    .setStatus(extras.getStatus())
                    .setDuration(extras.getDuration())
                    .setCode(extras.getCode());
        }
        if (entry.getTraceId() == null || entry.getTraceId().isEmpty()) {
            entry.setTraceId(TraceManager.getCurrentTraceId());
        }

        // ERROR 去重：相同 fingerprint 在 10s 窗口内只输出首条，窗口结束时补发汇总
        if (level == LogLevel.ERROR && shouldDedupError(entry)) {
            return;
        }

        emit(entry);
    }

    /** 实际输出日志条目：transport 输出 + 采样 + 限流 + 入队。去重汇总补发也走此路径 */
    private void emit(LogEntry entry) {
        // 逐条本地输出（Console/File transport），不受采样/限流影响
        for (LogTransport transport : transports) {
            transport.log(entry);
        }

        // 采样过滤
        Random random = new Random();
        if (random.nextInt(100) >= sampleRate(entry.getLevel())) {
            return;
        }
        // 限流
        if (!allowReport()) {
            return;
        }

        synchronized (queue) {
            if (queue.size() >= MAX_QUEUE) {
                queue.remove(0);
            }
            queue.add(entry);
            if (queue.size() >= FLUSH_THRESHOLD) {
                flush();
            }
        }
    }

    /**
     * ERROR 去重判定：相同 message + error_stack fingerprint 在 10s 窗口内只输出首条。
     * 窗口内重复命中累加计数并跳过输出；新 fingerprint 或窗口过期时补发上一轮汇总。
     * 返回 true 表示该条应被去重跳过，false 表示正常输出。
     */
    private boolean shouldDedupError(LogEntry entry) {
        int fingerprint = fingerprintHash(entry.getMessage(), entry.getErrorStack());
        long now = System.currentTimeMillis();
        boolean inWindow = lastErrorTime > 0 && (now - lastErrorTime) < DEDUP_WINDOW_MS;

        if (fingerprint == lastErrorFingerprint && inWindow) {
            errorDedupCount++;
            return true;
        }

        // 新 burst：先补发上一轮汇总（若有重复）
        flushDedupSummary();
        lastErrorFingerprint = fingerprint;
        lastErrorTime = now;
        errorDedupCount = 1;
        lastDedupEntry = entry;
        scheduleDedupSummary();
        return false;
    }

    /** 窗口结束时补发汇总条目：携带 dedupCount 标记本轮总次数，message 标注重复次数 */
    private synchronized void flushDedupSummary() {
        if (dedupSummaryFuture != null) {
            dedupSummaryFuture.cancel(false);
            dedupSummaryFuture = null;
        }
        int count = errorDedupCount;
        LogEntry original = lastDedupEntry;
        errorDedupCount = 0;
        lastErrorFingerprint = 0;
        lastErrorTime = 0L;
        lastDedupEntry = null;

        // 单次命中无重复时不补发，避免噪声
        if (count <= 1 || original == null) {
            return;
        }

        LogEntry summary = new LogEntry(LogLevel.ERROR,
                truncate(original.getMessage() + " (10s 内重复 " + (count - 1) + " 次)",
                        MAX_MESSAGE_LENGTH),
                app, appVersion)
                .setUrl(original.getUrl())
                .setUserAgent(original.getUserAgent())
                .setTraceId(TraceManager.getCurrentTraceId())
                .setErrorType(original.getErrorType())
                .setErrorSource(original.getErrorSource())
                .setErrorStack(original.getErrorStack())
                .setDedupCount(count);
        emit(summary);
    }

    private void scheduleDedupSummary() {
        if (dedupSummaryFuture != null) {
            dedupSummaryFuture.cancel(false);
        }
        dedupSummaryFuture = executor.schedule(this::flushDedupSummary,
                DEDUP_WINDOW_MS, TimeUnit.MILLISECONDS);
    }

    /** ERROR 去重 fingerprint：message + error_stack 的轻量 hash（djb2 变体，无需强 hash） */
    private static int fingerprintHash(String message, String errorStack) {
        String str = message + "|" + (errorStack != null ? errorStack : "");
        int hash = 0;
        for (int i = 0; i < str.length(); i++) {
            hash = hash * 31 + str.charAt(i);
        }
        return hash;
    }

    public void error(String message, LogEntry extras) {
        log(LogLevel.ERROR, message, extras);
    }

    public void warn(String message, LogEntry extras) {
        log(LogLevel.WARN, message, extras);
    }

    public void info(String message, LogEntry extras) {
        log(LogLevel.INFO, message, extras);
    }

    // ==================== 队列与上报 ====================

    private boolean allowReport() {
        long now = System.currentTimeMillis();
        Iterator<Long> it = sentTimestamps.iterator();
        while (it.hasNext()) {
            if (it.next() <= now - RATE_LIMIT_WINDOW_MS) {
                it.remove();
            } else {
                break;
            }
        }
        if (sentTimestamps.size() >= RATE_LIMIT_MAX) {
            return false;
        }
        sentTimestamps.add(now);
        return true;
    }

    public void flush() {
        RemoteTransport remote = findRemoteTransport();
        if (remote == null) {
            return;
        }
        synchronized (queue) {
            if (flushing || queue.isEmpty()) {
                return;
            }
            flushing = true;
            List<LogEntry> batch = new ArrayList<>(queue);
            queue.clear();
            try {
                remote.send(batch);
                backoffMs = INITIAL_BACKOFF_MS;
            } catch (Exception e) {
                // 上报失败：恢复队列，指数退避重试
                queue.addAll(0, batch);
                scheduleBackoff();
            } finally {
                flushing = false;
            }
        }
    }

    private RemoteTransport findRemoteTransport() {
        for (LogTransport transport : transports) {
            if (transport instanceof RemoteTransport) {
                return (RemoteTransport) transport;
            }
        }
        return null;
    }

    private void scheduleBackoff() {
        int delay = backoffMs;
        backoffMs = Math.min(backoffMs * 2, MAX_BACKOFF_MS);
        executor.schedule(this::flush, delay, TimeUnit.MILLISECONDS);
    }

    private static String truncate(String value, int max) {
        if (value == null) {
            return null;
        }
        return value.length() > max ? value.substring(0, max) : value;
    }

    /**
     * 启动时从本地文件补报（崩溃兜底 §3.5）。
     *
     * 读取 FileTransport 最近保留的 error.log，将日志条目重新入队并触发上报。
     * 适用于生产环境崩溃后重启：内存队列丢失，但 FileTransport 已持久化到磁盘。
     */
    public void flushFromDisk() {
        RemoteTransport remote = findRemoteTransport();
        if (remote == null) {
            return;
        }
        // 读取 FileTransport 持久化的 error.log，重新入队触发上报
        List<LogEntry> cached = new ArrayList<>();
        for (LogTransport transport : transports) {
            if (transport instanceof FileTransport) {
                try {
                    cached.addAll(((FileTransport) transport).readRecentErrorLogs(50));
                } catch (Exception ignored) {
                }
            }
        }
        if (cached.isEmpty()) {
            return;
        }
        synchronized (queue) {
            for (LogEntry entry : cached) {
                if (queue.size() >= MAX_QUEUE) {
                    break;
                }
                queue.add(entry);
            }
        }
        // 异步上报，避免在主线程（Application.onCreate）同步执行网络 IO 触发 ANR
        executor.execute(this::flush);
    }
}
