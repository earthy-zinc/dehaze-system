package com.pei.dehaze.sdk.model.task;

import lombok.Data;

/**
 * 任务VO
 * 对齐后端 TaskVO
 */
@Data
public class TaskVO {
    /** 任务主键ID */
    private long id;
    /** 任务ID（UUID） */
    private String taskId;
    /** 任务类型 */
    private String taskType;
    /** 任务状态：pending, processing, completed, failed, cancelled */
    private String status;
    /** 执行进度(0-100) */
    private int progress;
    /** 总文件数 */
    private int totalFiles;
    /** 已处理文件数 */
    private Integer processedFiles;
    /** 任务结果 */
    private String result;
    /** 下载链接 */
    private String downloadUrl;
    /** 错误信息 */
    private String error;
    /** 创建时间 */
    private String createdAt;
    /** 更新时间 */
    private String updatedAt;
    /** 开始时间 */
    private String startedAt;
    /** 完成时间 */
    private String completedAt;
    /** 过期时间 */
    private String expiresAt;
}
