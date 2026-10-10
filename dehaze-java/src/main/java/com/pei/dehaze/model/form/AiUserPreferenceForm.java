package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "用户 AI 偏好设置表单")
public class AiUserPreferenceForm {

    /** null 或缺失即清除偏好，故不加 @NotNull */
    @Schema(description = "偏好默认对话模型(model_id)；null 表示清除偏好")
    private String defaultModel;
}
