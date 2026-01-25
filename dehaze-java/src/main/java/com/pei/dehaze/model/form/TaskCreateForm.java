package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.util.List;

/**
 * 任务创建表单
 *
 * @author earthy-zinc
 * @since 2026-01-19
 */
@Schema(description = "任务创建表单")
@Data
public class TaskCreateForm {

    @Schema(
            description = "任务类型：dataset_export, item_download, batch_download",
            example = "dataset_export",
            allowableValues = {"dataset_export", "item_download", "batch_download"}
    )
    @NotBlank(message = "任务类型不能为空")
    private String type;

    @Schema(description = "目标ID（单个资源）", example = "123")
    private Long targetId;

    @Schema(description = "目标ID列表（批量操作）", example = "[1, 2, 3]")
    private List<Long> targetIds;

    @Schema(description = "任务选项")
    private TaskOptions options = new TaskOptions();

    /**
     * 任务选项
     */
    @Data
    @Schema(description = "任务选项")
    public static class TaskOptions {

        @Schema(
                description = "文件组织方式：by_item（按数据项）, flat（扁平结构）",
                example = "by_item",
                allowableValues = {"by_item", "flat"}
        )
        private String structure = "by_item";

        @Schema(
                description = "包含类型：clear（清晰图）, hazy（有雾图）",
                example = "[\"clear\", \"hazy\"]"
        )
        private List<String> includeTypes;

        @Schema(description = "是否包含缩略图", example = "false")
        private Boolean includeThumbnail = false;
    }
}
