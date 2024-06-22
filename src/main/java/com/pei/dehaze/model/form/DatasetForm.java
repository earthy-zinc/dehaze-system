package com.pei.dehaze.model.form;

import com.pei.dehaze.common.validator.PathExists;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "数据集表单对象")
@Data
public class DatasetForm {
    @Schema(description = "数据集ID")
    private Long id;

    @Schema(description = "父数据集ID")
    @NotNull(message = "父数据集ID不能为空")
    private Long parentId;

    @Schema(description = "数据集类型")
    private String type;

    @Schema(description = "数据集名称")
    private String name;

    @Schema(description = "数据集描述")
    private String description;

    @Schema(description = "数据集存储路径")
    @PathExists
    private String path;

    @Schema(description = "状态(1:正常;0:禁用)")
    private Integer status;
}
