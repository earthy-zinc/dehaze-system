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

    @Schema(description = "任务状态：pending, processing, completed, failed, cancelled", example = "pending")
    private String status;

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
}
