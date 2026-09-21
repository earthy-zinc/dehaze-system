package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "Skill 共享至市场表单")
@Data
public class SkillShareForm {

    @NotNull(message = "Skill ID 不能为空")
    private Long skillId;
}
