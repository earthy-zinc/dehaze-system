package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotEmpty;
import lombok.Data;

import java.util.List;

/**
 * 批量下载数据项表单
 *
 * @author earthy-zinc
 * @since 2025-12-13
 */
@Data
@Schema(description = "批量下载数据项表单")
public class BatchDownloadForm {

    @NotEmpty(message = "请选择要下载的数据项")
    @Schema(
            description = "要下载的数据项文件ID列表，支持批量下载多个图片文件",
            requiredMode = Schema.RequiredMode.REQUIRED,
            example = "[1, 2, 3]"
    )
    private List<Long> itemFileIds;

    @Schema(
            description = "是否按数据项分目录组织，true-按数据项分目录，false-扁平结构",
            example = "true"
    )
    private Boolean organizeByItem = true;
}
