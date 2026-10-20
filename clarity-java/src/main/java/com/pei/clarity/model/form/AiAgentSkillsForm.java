package com.pei.clarity.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;
import lombok.Data;

/**
 * 设置 Agent Skill 请求
 *
 * @author clarity
 */
@Data
public class AiAgentSkillsForm {

    @Schema(description = "Skill 名称列表(覆盖式更新)")
    private List<String> skills;
}
