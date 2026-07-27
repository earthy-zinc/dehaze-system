package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

/**
 * 导出任务 VO（异步导出时返回）
 */
@Data
@Builder
@Schema(description = "导出任务信息")
public class ExportTaskVO {

    @Schema(description = "任务ID")
    private String taskId;

    @Schema(description = "任务状态：1=待处理,2=处理中,3=已完成,4=失败,5=已取消")
    private Integer status;

    @Schema(description = "预估数据条数")
    private Long estimatedCount;
}
