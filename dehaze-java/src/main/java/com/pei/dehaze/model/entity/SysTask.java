package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

/**
 * 系统任务实体
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Data
@EqualsAndHashCode(callSuper = true)
@TableName("sys_task")
public class SysTask extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String taskId;

    private String taskType;

    private Integer status;

    private Integer progress;

    private Integer totalFiles;

    private Integer processedFiles;

    private String params;

    private String result;

    private String errorMessage;

    private LocalDateTime startedAt;

    private LocalDateTime completedAt;

    private LocalDateTime expiresAt;

    /** 客户端幂等键（HTTP Idempotency-Key 头） */
    private String idempotencyKey;

    /** MQ 重试次数 */
    private Integer retryCount;

    /** 执行 Worker 标识 */
    private String workerId;
}
