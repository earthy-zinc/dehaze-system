package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 任务VO（通用任务响应）
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Data
@Schema(description = "任务信息")
public class TaskVO {

    @Schema(description = "任务ID", example = "uuid-xxx")
    private String taskId;

    @Schema(description = "任务类型：dataset_export, user_export, role_export, dept_export, menu_export, dict_export, algorithm_export, user_import 等", example = "dataset_export")
    private String taskType;

    @Schema(description = "任务类别：import / export", example = "export")
    private String taskCategory;

    @Schema(description = "任务状态：1=待处理,2=处理中,3=已完成,4=失败,5=已取消", example = "1")
    private Integer status;

    @Schema(description = "进度（0-100）", example = "0")
    private Integer progress;

    @Schema(description = "文件总数", example = "100")
    private Integer totalFiles;

    @Schema(description = "已处理文件数", example = "0")
    private Integer processedFiles;

    @Schema(description = "下载链接")
    private String downloadUrl;

    @Schema(description = "过期时间")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime expiresAt;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime createdAt;

    @Schema(description = "开始时间")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime startedAt;

    @Schema(description = "完成时间")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime completedAt;

    @Schema(description = "错误信息")
    private String error;

    @Schema(description = "客户端幂等键")
    private String idempotencyKey;

    @Schema(description = "MQ 重试次数")
    private Integer retryCount;

    @Schema(description = "执行 Worker 标识")
    private String workerId;
}
