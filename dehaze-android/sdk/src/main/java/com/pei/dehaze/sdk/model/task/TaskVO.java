package com.pei.dehaze.sdk.model.task;

import lombok.Data;

/**
 * 任务VO
 * 对齐后端 TaskVO
 */
@Data
public class TaskVO {
    /** 任务ID（UUID） */
    private String taskId;
    /** 任务类型 */
    private TaskType taskType;
    /** 任务类别：import/export */
    private String taskCategory;
    /** 任务状态 */
    private TaskStatus status;
    /** 执行进度(0-100) */
    private int progress;
    /** 总文件数 */
    private int totalFiles;
    /** 已处理文件数 */
    private Integer processedFiles;
    /** 下载链接（指向存储后端 URL） */
    private String downloadUrl;
    /** 错误信息 */
    private String error;
    /** 创建时间 */
    private String createdAt;
    /** 开始时间 */
    private String startedAt;
    /** 完成时间 */
    private String completedAt;
    /** 过期时间 */
    private String expiresAt;
    /** 客户端幂等键 */
    private String idempotencyKey;
    /** MQ 重试次数 */
    private Integer retryCount;
    /** 执行 Worker 标识 */
    private String workerId;
}
