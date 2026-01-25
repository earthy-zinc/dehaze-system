package com.pei.dehaze.model.form;

import com.pei.dehaze.common.enums.StatusEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Schema(description = "数据集更新表单")
@Data
public class DatasetUpdateForm {

    @Schema(
            description = "数据集类型：样例数据集、测试数据集、用户数据集、处理结果集",
            example = "用户数据集"
    )
    private String type;

    @Schema(
            description = "数据集名称，同一父数据集下名称需唯一",
            example = "我的测试数据集"
    )
    private String name;

    @Schema(
            description = "数据集描述信息",
            example = "用于测试去雾算法的数据集"
    )
    private String description;

    @Schema(
            description = "数据集状态：1-启用，0-禁用",
            example = "ENABLE",
            allowableValues = {"1", "0"}
    )
    private StatusEnum status;
}
