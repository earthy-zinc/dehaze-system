package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

/**
 * 导入任务 VO（异步导入时返回）
 */
@Data
@Builder
@Schema(description = "导入任务信息")
public class ImportTaskVO {

    @Schema(description = "任务ID")
    private String taskId;

    @Schema(description = "任务状态：PENDING / PROCESSING / COMPLETED / FAILED / CANCELLED")
    private String status;
}
