package com.pei.dehaze.model.form;

import com.pei.dehaze.common.enums.StatusEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Schema(description = "数据集创建表单")
@Data
public class DatasetAddForm {

    @Schema(
            description = "父数据集ID，0表示根数据集",
            requiredMode = Schema.RequiredMode.REQUIRED,
            example = "0"
    )
    @NotNull(message = "父数据集ID不能为空")
    private Long parentId;

    @Schema(
            description = "数据集类型：样例数据集、测试数据集、用户数据集、处理结果集",
            example = "用户数据集"
    )
    @NotBlank(message = "数据集类型不能为空")
    private String type;

    @Schema(
            description = "数据集名称，同一父数据集下名称需唯一",
            example = "我的测试数据集"
    )
    @NotBlank(message = "数据集名称不能为空")
    @Size(min = 1, max = 255, message = "数据集名称长度必须在1-255之间")
    private String name;

    @Schema(
            description = "数据集描述信息",
            example = "用于测试去雾算法的数据集"
    )
    @Size(max = 500, message = "描述信息长度不能超过500字符")
    private String description;

    @Schema(
            description = "数据集状态：1-启用，0-禁用",
            example = "1",
            allowableValues = {"0", "1"}
    )
    private StatusEnum status;
}
