package com.pei.dehaze.model.form;

import com.pei.dehaze.common.validator.PathExists;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "算法表单对象")
@Data
public class AlgorithmForm {
    @Schema(description = "算法ID")
    private Long id;

    @Schema(description = "算法父级ID，置0为顶级算法")
    @NotNull(message = "父算法ID不能为空")
    private Long parentId;

    @Schema(description = "算法类型")
    private String type;

    @Schema(description = "算法名称")
    private String name;

    @Schema(description = "算法模型文件存储路径")
    @PathExists
    private String path;

    @Schema(description = "算法代码Python导入路径")
    private String importPath;

    @Schema(description = "算法描述")
    private String description;

    /**
     * 状态(1:正常;0:禁用)
     */
    @Schema(description = "算法状态(1:正常;0:禁用)")
    private Integer status;

}
