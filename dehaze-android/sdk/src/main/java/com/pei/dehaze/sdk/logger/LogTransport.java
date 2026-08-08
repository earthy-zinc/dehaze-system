package com.pei.dehaze.sdk.logger;

import java.util.List;

/**
 * 日志 transport 接口（多 transport 架构，§3.6）。
 */
public interface LogTransport {

    /** 逐条本地输出（不受采样/限流影响）。 */
    void log(LogEntry entry);

    /** 批量上报（仅 RemoteTransport 实现，其余空实现）。 */
    default void send(List<LogEntry> logs) throws Exception {
    }
}
