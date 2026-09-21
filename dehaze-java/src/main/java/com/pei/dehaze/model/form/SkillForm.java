package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Schema(description = "Skill 创建表单")
@Data
public class SkillForm {

    @NotBlank(message = "Skill 名称不能为空")
    @Size(max = 128)
    private String name;

    @NotBlank(message = "Skill 描述不能为空")
    @Size(max = 500)
    private String description;

    @Size(max = 255)
    private String scene = "";

    @NotBlank(message = "Skill 指令不能为空")
    @Schema(description = "Markdown 指令全文")
    private String instruction;
}
