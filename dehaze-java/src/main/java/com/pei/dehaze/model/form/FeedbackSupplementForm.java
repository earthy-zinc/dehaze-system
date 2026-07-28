package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "反馈补充说明表单")
public class FeedbackSupplementForm {

    @Schema(description = "补充内容", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "补充内容不能为空")
    private String content;

    @Schema(description = "附件URL")
    private List<String> attachments;
}
