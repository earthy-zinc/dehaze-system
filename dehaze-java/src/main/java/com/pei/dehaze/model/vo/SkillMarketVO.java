package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Schema(description = "Skill 市场目录项")
@Data
public class SkillMarketVO {

    private Long skillId;

    private String name;

    private String description;

    private String scene;

    private Boolean enabled;

    private Integer agentCount;
}
