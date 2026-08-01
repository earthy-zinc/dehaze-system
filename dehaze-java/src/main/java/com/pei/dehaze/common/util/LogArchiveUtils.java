package com.pei.dehaze.common.util;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.util.List;
import java.util.stream.Stream;

/**
 * 日志启动归档工具：dev 环境重启时把当天已存在的活动日志归档为 {级别}.{n}.log，
 * 使本次启动日志写入全新文件。须在 logback 打开文件前调用。
 */
public final class LogArchiveUtils {

    private static final String LOG_HOME = "./logs";
    private static final String[] LOG_LEVELS = {"info", "error"};

    private LogArchiveUtils() {
    }

    /**
     * 归档当天已存在且非空的活动日志文件为 {level}.{n}.log，
     * 与 logback SizeAndTimeBasedRollingPolicy 的分片命名保持一致。
     */
    public static void archiveTodayLogs() {
        Path dir = Paths.get(LOG_HOME, LocalDate.now().toString());
        if (!Files.isDirectory(dir)) {
            return;
        }
        for (String level : LOG_LEVELS) {
            Path active = dir.resolve(level + ".log");
            File file = active.toFile();
            if (!file.exists() || file.length() == 0) {
                continue;
            }
            Path archived = dir.resolve(level + "." + nextIndex(dir, level) + ".log");
            try {
                Files.move(active, archived);
            } catch (Exception ignored) {
                // 归档失败不影响启动，继续追加原文件
            }
        }
    }

    /** 在 dir 下找 {level}.{n}.log 的最大序号 +1（活动文件 {level}.log 不计入） */
    private static int nextIndex(Path dir, String level) {
        int n = 0;
        String prefix = level + ".";
        int suffixLen = ".log".length();
        try (Stream<Path> stream = Files.list(dir)) {
            List<Path> files = stream.toList();
            for (Path p : files) {
                String name = p.getFileName().toString();
                if (!name.startsWith(prefix) || !name.endsWith(".log")) {
                    continue;
                }
                // 活动文件 info.log 无数字段（长度 == prefix + ".log"），跳过避免 substring 越界
                if (name.length() <= prefix.length() + suffixLen) {
                    continue;
                }
                String num = name.substring(prefix.length(), name.length() - suffixLen);
                try {
                    int i = Integer.parseInt(num);
                    if (i > n) {
                        n = i;
                    }
                } catch (NumberFormatException ignored) {
                }
            }
        } catch (Exception ignored) {
        }
        return n + 1;
    }
}
