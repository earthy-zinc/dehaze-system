package com.pei.dehaze.sdk.model.task;

import java.util.List;

import lombok.Data;

/**
 * 导出任务创建表单
 * 对齐后端 ExportTaskCreateForm
 */
@Data
public class TaskCreateForm {
    /** 导出类型 */
    private TaskType type;
    /** 单个导出目标ID（type为dataset_export或item_download时使用） */
    private Long targetId;
    /** 批量导出目标ID列表（type为batch_download或custom_export时使用） */
    private List<Long> targetIds;
    /** 导出选项 */
    private ExportOptions options;
}
