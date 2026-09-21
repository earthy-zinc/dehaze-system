package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Schema(description = "Skill 更新表单")
@Data
public class SkillUpdateForm {

    @Size(min = 1, max = 128)
    private String name;

    @Size(min = 1, max = 500)
    private String description;

    @Size(max = 255)
    private String scene;

    @Size(min = 1)
    private String instruction;
}
