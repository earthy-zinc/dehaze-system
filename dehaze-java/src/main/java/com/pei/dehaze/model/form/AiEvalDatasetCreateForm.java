package com.pei.dehaze.model.form;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * 创建评测集请求
 *
 * @author dehaze
 */
@Data
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class AiEvalDatasetCreateForm {

    @NotBlank(message = "评测集名称不能为空")
    @Size(max = 128, message = "评测集名称长度不能超过128")
    @Schema(description = "评测集名称")
    private String name;

    @Size(max = 512, message = "评测集描述长度不能超过512")
    @Schema(description = "评测集描述")
    private String description;

    @NotBlank(message = "评测集类型不能为空")
    @Schema(description = "评测集类型(dev/regression/heldout)")
    private String datasetType;

}
