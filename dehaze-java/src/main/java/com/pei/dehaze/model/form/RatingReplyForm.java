package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
@Schema(description = "评价回复表单")
public class RatingReplyForm {

    @Schema(description = "回复内容", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "回复内容不能为空")
    private String content;
}
