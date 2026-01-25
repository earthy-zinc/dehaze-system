package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 系统任务实体
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Data
@TableName("sys_task")
public class SysTask {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String taskId;

    private String taskType;

    private String status;

    private Integer progress;

    private Integer totalFiles;

    private Integer processedFiles;

    private String params;

    private String result;

    private String errorMessage;

    private Long createdBy;

    private LocalDateTime createdAt;

    private LocalDateTime startedAt;

    private LocalDateTime completedAt;

    private LocalDateTime expiresAt;
}
