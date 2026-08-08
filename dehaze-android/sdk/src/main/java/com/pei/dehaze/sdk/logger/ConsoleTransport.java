package com.pei.dehaze.sdk.logger;

/**
 * 开发环境 transport：输出到控制台（logcat / System.out）。
 */
public class ConsoleTransport implements LogTransport {

    @Override
    public void log(LogEntry entry) {
        String tag = "[dehaze][" + entry.getLevel().getLabel() + "]";
        String message = entry.getMessage() + " trace_id=" +
                (entry.getTraceId() != null ? entry.getTraceId() : "");
        String line;
        if (entry.getLevel() == LogLevel.ERROR) {
            line = tag + " " + message + "\n" +
                    (entry.getErrorStack() != null ? entry.getErrorStack() : "");
        } else {
            line = tag + " " + message;
        }
        System.out.println("DehazeLog: " + line);
    }
}
