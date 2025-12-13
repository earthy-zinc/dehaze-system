package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

/**
 *
 * @author earthy-zinc
 * @since 2025-12-13 14:05:29
 */
@Data
@Schema(description = "图片上传表单")
public class ItemFileUploadForm {

    @Schema(description = "表单文件对象", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "文件不能为空")
    private MultipartFile file;

    @Schema(description = "所属数据集ID", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "数据集ID不能为空")
    private Long datasetId;

    @Schema(description = "所属数据项ID", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "数据项ID不能为空")
    private Long datasetItemId;

    @Schema(description = "图片类型", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "图片类型不能为空")
    private String type;

    @Schema(description = "图片描述")
    private String description;

    @Schema(description = "场景类型")
    private String sceneType;

    @Schema(description = "雾霾程度")
    private String hazeLevel;
}
