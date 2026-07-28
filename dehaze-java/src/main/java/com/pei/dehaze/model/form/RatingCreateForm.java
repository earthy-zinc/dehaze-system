package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "评价创建/修改表单")
public class RatingCreateForm {

    @Schema(description = "处理记录ID", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "处理记录ID不能为空")
    private Long predLogId;

    @Schema(description = "评分(1-5)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "评分不能为空")
    @Min(value = 1, message = "评分不能小于1")
    @Max(value = 5, message = "评分不能大于5")
    private Integer rating;

    @Schema(description = "评价文字(最多500字符)")
    private String comment;

    @Schema(description = "评价标签")
    private List<String> tags;

    @Schema(description = "截图URL(最多3张)")
    private List<String> imageUrls;

    @Schema(description = "是否匿名(0:否;1:是)")
    private Integer isAnonymous;
}
