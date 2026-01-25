package com.pei.dehaze.service;

import com.pei.dehaze.model.form.ExportTaskCreateForm;

/**
 * 任务执行器接口（独立类，修复异步失效问题）
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
public interface TaskExecutor {

    /**
     * 提交导出任务到异步执行器
     *
     * @param taskId 数据库任务ID
     * @param form   导出任务创建表单
     */
    void submitExportTask(Long taskId, ExportTaskCreateForm form);
}
