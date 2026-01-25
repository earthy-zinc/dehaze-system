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
            description = "任务类型：dataset_export-数据集导出, item_download-数据项下载, batch_download-批量下载",
            example = "dataset_export"
    )
    private String taskType;

    @Schema(
            description = "任务状态：pending-等待中, processing-处理中, completed-已完成, failed-失败, cancelled-已取消",
            example = "processing"
    )
    private String status;
}
