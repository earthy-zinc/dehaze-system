package com.pei.dehaze.sdk.logger;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * 本地文件 transport（开发+生产崩溃兜底）。
 *
 * 目录结构 `logs/{yyyy-MM-dd}/{level}.log`（NDJSON），单文件 100MB 归档为
 * `{level}.{n}.log`，超过 retentionDays 的日期目录自动清理。
 * 根目录由初始化时传入（Android 端传 context.filesDir）。
 */
public class FileTransport implements LogTransport {

    private static final long MAX_FILE_BYTES = 100L * 1024 * 1024; // 100MB

    private final File rootDir;
    private final int retentionDays;
    private boolean cleanedToday = false;

    public FileTransport(File rootDir, int retentionDays) {
        this.rootDir = new File(rootDir, "logs");
        this.retentionDays = retentionDays;
    }

    public File getRootDir() {
        return rootDir;
    }

    @Override
    public void log(LogEntry entry) {
        String dateDir = formatDate(new Date());
        File dir = new File(rootDir, dateDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        cleanupOldDirs(dateDir);

        String line = entry.toJson().toString() + "\n";
        // info.log 包含 INFO+（全部级别）
        File infoTarget = rotateIfNeeded(dir, "info");
        try (FileWriter writer = new FileWriter(infoTarget, true)) {
            writer.write(line);
        } catch (IOException ignored) {
        }
        // error.log 仅 ERROR
        if (entry.getLevel() == LogLevel.ERROR) {
            File errorTarget = rotateIfNeeded(dir, "error");
            try (FileWriter writer = new FileWriter(errorTarget, true)) {
                writer.write(line);
            } catch (IOException ignored) {
            }
        }
    }

    private String formatDate(Date date) {
        return new SimpleDateFormat("yyyy-MM-dd", Locale.US).format(date);
    }

    private void cleanupOldDirs(String currentDate) {
        if (cleanedToday) {
            return;
        }
        cleanedToday = true;
        File[] dirs = rootDir.listFiles(File::isDirectory);
        if (dirs == null) {
            return;
        }
        Calendar cutoff = Calendar.getInstance();
        cutoff.add(Calendar.DAY_OF_YEAR, -retentionDays);
        for (File dir : dirs) {
            try {
                Date date = new SimpleDateFormat("yyyy-MM-dd", Locale.US)
                        .parse(dir.getName());
                if (date != null && date.before(cutoff.getTime())) {
                    deleteRecursively(dir);
                }
            } catch (Exception ignored) {
                // 目录名不是日期，跳过
            }
        }
    }

    private static void deleteRecursively(File file) {
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) {
                deleteRecursively(child);
            }
        }
        file.delete();
    }

    private File rotateIfNeeded(File dir, String level) {
        File current = new File(dir, level + ".log");
        if (current.exists() && current.length() >= MAX_FILE_BYTES) {
            // 归档为 {level}.{n}.log（n 递增）
            int n = 1;
            while (new File(dir, level + "." + n + ".log").exists()) {
                n++;
            }
            current.renameTo(new File(dir, level + "." + n + ".log"));
            return new File(dir, level + ".log");
        }
        return current;
    }

    /**
     * 读取最近 error.log 中的日志条目（供启动补报使用）。
     *
     * @param limit 最多读取条数
     */
    public List<LogEntry> readRecentErrorLogs(int limit) {
        List<LogEntry> result = new ArrayList<>();
        String dateDir = formatDate(new Date());
        File dir = new File(rootDir, dateDir);
        File errorLog = new File(dir, "error.log");
        if (!errorLog.exists()) {
            return result;
        }
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(errorLog), StandardCharsets.UTF_8))) {
            // 只读取最后 limit 行（倒序读取）
            List<String> lines = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
            int start = Math.max(0, lines.size() - limit);
            for (int i = start; i < lines.size(); i++) {
                try {
                    LogEntry entry = parseNdjson(lines.get(i));
                    if (entry != null) {
                        result.add(entry);
                    }
                } catch (Exception ignored) {
                }
            }
        } catch (IOException ignored) {
        }
        return result;
    }

    private LogEntry parseNdjson(String line) {
        if (line == null || line.trim().isEmpty()) {
            return null;
        }
        JsonObject obj = JsonParser.parseString(line.trim()).getAsJsonObject();
        String levelStr = obj.has("level") ? obj.get("level").getAsString() : "INFO";
        LogLevel level = "ERROR".equals(levelStr) ? LogLevel.ERROR
                : "WARN".equals(levelStr) ? LogLevel.WARN : LogLevel.INFO;
        String message = obj.has("message") ? obj.get("message").getAsString() : "";
        LogEntry entry = new LogEntry(level, message, "", "");
        if (obj.has("trace_id")) entry.setTraceId(obj.get("trace_id").getAsString());
        if (obj.has("error_type")) entry.setErrorType(obj.get("error_type").getAsString());
        if (obj.has("error_source")) entry.setErrorSource(obj.get("error_source").getAsString());
        if (obj.has("error_stack")) entry.setErrorStack(obj.get("error_stack").getAsString());
        if (obj.has("method")) entry.setMethod(obj.get("method").getAsString());
        if (obj.has("path")) entry.setPath(obj.get("path").getAsString());
        if (obj.has("status")) entry.setStatus(obj.get("status").getAsInt());
        if (obj.has("duration")) entry.setDuration(obj.get("duration").getAsDouble());
        if (obj.has("code")) entry.setCode(obj.get("code").getAsString());
        return entry;
    }
}
