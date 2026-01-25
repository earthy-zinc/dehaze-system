package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

/**
 * 导出任务请求
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@Schema(description = "导出任务请求")
@Data
public class ExportTaskRequest {

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
