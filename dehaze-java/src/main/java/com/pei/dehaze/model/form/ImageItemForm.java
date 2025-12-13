package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 图片信息修改表单（包含标注信息）
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
@Data
@Schema(description = "图片信息修改表单")
public class ImageItemForm {

    @Schema(description = "数据项文件ID", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "数据项文件ID不能为空")
    private Long itemFileId;

    @Schema(description = "图片类型")
    private String type;

    @Schema(description = "场景类型")
    private String sceneType;

    @Schema(description = "雾霾程度")
    private String hazeLevel;

    @Schema(description = "图片描述")
    private String description;
}
