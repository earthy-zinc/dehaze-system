package com.pei.dehaze.model.form;

import com.pei.dehaze.common.validator.FileExists;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
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
    @NotBlank(message = "算法类型不能为空")
    private String type;

    @Schema(description = "算法名称")
    @NotBlank(message = "算法名称不能为空")
    @Size(max = 50, message = "算法名称长度不能超过50")
    private String name;

    @Schema(description = "算法模型文件存储路径")
    @FileExists
    private String path;

    @Schema(description = "算法代码Python导入路径")
    private String importPath;

    @Schema(description = "算法描述")
    private String description;

    @Schema(description = "算法状态(1:草稿;2:测试中;3:待审核;4:已发布;5:已停用;6:已归档)")
    private Integer status;

}
