package com.pei.dehaze.service;

import com.pei.dehaze.model.form.ExportTaskCreateForm;

/**
 * 任务执行器接口（MQ 统一路径）
 *
 * <p>发布任务到 RabbitMQ，由 MQ Consumer 调用 {@link #executeExportTask} 执行。
 * MQ 未启用时（测试环境）fallback 到同步执行。
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
public interface TaskExecutor {

    /**
     * 发布导出任务到 MQ
     *
     * @param taskId 数据库任务ID
     */
    void publishExportTask(Long taskId);

    /**
     * 执行导出任务（由 MQ Consumer 调用，或测试直接调用）
     *
     * @param taskId 数据库任务ID
     * @param form   导出任务创建表单
     */
    void executeExportTask(Long taskId, ExportTaskCreateForm form);
}
