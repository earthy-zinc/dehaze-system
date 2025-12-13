package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/**
 * 批量图片上传表单
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
@Data
@Schema(description = "批量图片上传表单")
public class BatchDatasetItemUploadForm {

    @Schema(description = "数据集ID")
    @NotNull(message = "数据集ID不能为空")
    private Long datasetId;

    @Schema(description = "文件列表")
    @NotEmpty(message = "文件列表不能为空")
    private List<MultipartFile> files;

    @Schema(description = "场景类型（可选，应用于所有配对）")
    private String sceneType;
}
