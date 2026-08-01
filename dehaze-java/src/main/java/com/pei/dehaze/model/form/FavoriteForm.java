package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
@Schema(description = "添加收藏表单")
public class FavoriteForm {

    @Schema(description = "收藏对象类型(algorithm/result/dataset/image/preset)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "收藏对象类型不能为空")
    private String targetType;

    @Schema(description = "收藏对象ID", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "收藏对象ID不能为空")
    private Long targetId;
}
