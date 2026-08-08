package com.pei.dehaze.service;

import com.pei.dehaze.model.form.ClientLogBatchForm;

/**
 * 前端日志接收服务。
 * <p>
 * 将前端 SDK 批量上报的日志以 NDJSON 形式写入 logs/{yyyy-MM-dd}/client.log，
 * 供 filebeat 采集进入 ELK，与后端日志通过 trace_id 串联。
 */
public interface ClientLogService {

    /**
     * 批量接收并落盘前端日志。
     *
     * @param form 前端日志批量上报请求体
     */
    void collect(ClientLogBatchForm form);
}
