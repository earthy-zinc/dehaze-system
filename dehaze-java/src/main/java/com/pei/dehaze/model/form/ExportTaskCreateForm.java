package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;

/**
 * 导出任务创建表单
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Schema(description = "导出任务创建表单")
@Data
public class ExportTaskCreateForm {

    @Schema(
            description = "导出类型",
            example = "dataset",
            allowableValues = {"dataset", "dataset_item", "batch_items", "custom"}
    )
    @NotBlank(message = "导出类型不能为空")
    private String type;

    @Schema(
            description = "目标ID（导出单个资源时使用）",
            example = "123"
    )
    private Long targetId;

    @Schema(
            description = "目标ID列表（批量导出时使用）",
            example = "[1, 2, 3]"
    )
    private List<Long> targetIds;

    @Schema(description = "导出选项配置")
    private ExportOptions options = new ExportOptions();

    /**
     * 导出选项配置
     */
    @Data
    @Schema(description = "导出选项配置")
    public static class ExportOptions {

        @Schema(
                description = "文件组织方式：by_item（按数据项）, flat（扁平结构）",
                example = "by_item",
                allowableValues = {"by_item", "flat"}
        )
        private String structure = "by_item";

        @Schema(
                description = "包含的类型：clear（清晰图）, hazy（有雾图）",
                example = "[\"clear\", \"hazy\"]"
        )
        private List<String> includeTypes;

        @Schema(
                description = "是否包含缩略图",
                example = "false"
        )
        private Boolean includeThumbnail = false;
    }
}
