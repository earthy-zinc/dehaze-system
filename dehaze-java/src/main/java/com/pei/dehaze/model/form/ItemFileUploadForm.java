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
@Schema(description = "数据项图片上传表单")
public class ItemFileUploadForm {

    @Schema(
            description = "图片文件，支持jpg/png/gif格式，单张不超过10MB",
            requiredMode = Schema.RequiredMode.REQUIRED
    )
    @NotNull(message = "文件不能为空")
    private MultipartFile file;

    @Schema(
            description = "所属数据项ID，指定图片归属的数据项",
            requiredMode = Schema.RequiredMode.REQUIRED,
            example = "1"
    )
    @NotNull(message = "数据项ID不能为空")
    private Long itemId;

    @Schema(
            description = "图片类型：clear(清晰图/GT) / hazy(有雾图) / trans(透射率图) / depth(深度图) / segment(分割图)",
            requiredMode = Schema.RequiredMode.REQUIRED,
            example = "hazy"
    )
    @NotBlank(message = "图片类型不能为空")
    private String type;

    @Schema(
            description = "图片描述信息",
            example = "城市街道场景的有雾图像"
    )
    private String description;

    @Schema(
            description = "场景类型，用户自定义的场景分类（如：城市街道、山区风景、海边等）",
            example = "城市街道"
    )
    private String sceneType;

    @Schema(
            description = "雾霾程度，支持多种规范：light/medium/heavy（人工分级）、beta=0.5（β参数）、A=0.8,beta=0.2（A+β双参数），可为空",
            example = "medium"
    )
    private String hazeLevel;
}
