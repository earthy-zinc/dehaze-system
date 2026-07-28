package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "反馈创建表单")
public class FeedbackCreateForm {

    @Schema(description = "反馈类型(suggestion/bug/experience/complaint)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "反馈类型不能为空")
    private String feedbackType;

    @Schema(description = "反馈标题(5-50字符)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "标题不能为空")
    private String title;

    @Schema(description = "反馈内容(10-1000字符)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "反馈内容不能为空")
    private String content;

    @Schema(description = "联系方式")
    private String contact;

    @Schema(description = "截图URL(最多5张)")
    private List<String> images;

    @Schema(description = "相关模块")
    private String relatedModule;
}
