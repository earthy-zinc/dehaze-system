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
            description = "任务类型：dataset_export-数据集导出, item_download-数据项下载, batch_download-批量下载, custom_export-自定义导出",
            example = "dataset_export"
    )
    private String taskType;

    @Schema(
            description = "任务状态：PENDING-等待中, PROCESSING-处理中, COMPLETED-已完成, FAILED-失败, CANCELLED-已取消",
            example = "PROCESSING"
    )
    private String status;
}
