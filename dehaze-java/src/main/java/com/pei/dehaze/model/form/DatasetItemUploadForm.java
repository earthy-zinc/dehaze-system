package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import jakarta.validation.constraints.NotNull;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/**
 * 配对图片上传表单
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
@Data
@Schema(description = "配对图片上传表单")
public class DatasetItemUploadForm {

    @Schema(description = "数据集ID")
    @NotNull(message = "数据集ID不能为空")
    private Long datasetId;

    @Schema(description = "数据项名称")
    private String name;

    @Schema(description = "清晰图文件（可选，适配不同数据集规范）")
    private MultipartFile clearImage;

    @Schema(description = "有雾图文件列表（可选，适配不同数据集规范）")
    private List<MultipartFile> hazyImages;

    @Schema(description = "对应的雾霾程度列表（可选，支持多种规范：light/medium/heavy、beta=X、A=X,beta=Y 等）")
    private List<String> hazeLevels;

    @Schema(description = "场景类型")
    private String sceneType;
}
