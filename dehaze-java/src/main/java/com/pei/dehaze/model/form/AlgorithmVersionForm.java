package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * 算法版本创建表单
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "算法版本创建表单")
@Data
public class AlgorithmVersionForm {

    @Schema(description = "版本号（语义化版本，如 1.0.1）")
    @NotBlank(message = "版本号不能为空")
    private String version;

    @Schema(description = "变更日志")
    private String changeLog;

    @Schema(description = "模型文件ID")
    private Long modelFileId;
}
