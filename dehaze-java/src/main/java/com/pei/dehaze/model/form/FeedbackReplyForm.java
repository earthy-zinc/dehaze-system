package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "反馈回复表单")
public class FeedbackReplyForm {

    @Schema(description = "回复内容", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "回复内容不能为空")
    private String content;

    @Schema(description = "回复类型(info/resolved/unsupported/dev_transfer)")
    private String replyType;

    @Schema(description = "附件URL")
    private List<String> attachments;
}
