package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "反馈创建表单")
public class FeedbackCreateForm {

    @Schema(description = "反馈类型(suggestion/bug/experience/complaint)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "反馈类型不能为空")
    @Pattern(regexp = "^(suggestion|bug|experience|complaint)$", message = "非法的反馈类型")
    private String feedbackType;

    @Schema(description = "反馈标题(5-50字符)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "标题不能为空")
    @Size(min = 5, max = 50, message = "标题长度必须在5-50字符之间")
    private String title;

    @Schema(description = "反馈内容(10-1000字符)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "反馈内容不能为空")
    @Size(min = 10, max = 1000, message = "内容长度必须在10-1000字符之间")
    private String content;

    @Schema(description = "联系方式")
    private String contact;

    @Schema(description = "截图URL(最多5张)")
    private List<String> images;

    @Schema(description = "相关模块")
    private String relatedModule;
}
