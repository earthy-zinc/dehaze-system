package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 任务查询参数
 *
 * @author earthy-zinc
 * @since 2025-01-19
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "任务查询参数")
public class TaskQuery extends BasePageQuery {

    @Schema(
            description = "任务类型（支持逗号分隔多个）：如 dataset_export, user_export, user_import 等",
            example = "user_export"
    )
    private String taskType;

    @Schema(
            description = "任务状态：PENDING-等待中, PROCESSING-处理中, COMPLETED-已完成, FAILED-失败, CANCELLED-已取消",
            example = "PROCESSING"
    )
    private String status;

    @Schema(
            description = "任务类别筛选：import-导入, export-导出",
            example = "export"
    )
    private String taskCategory;
}
