package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/** Skill 列表项（不含指令全文，渐进式加载不注入 instruction） */
@Schema(description = "Skill 列表项")
@Data
public class SkillListItemVO {

    private Long id;

    private String name;

    private String description;

    private String scene;

    private Integer status;

    private String source;

    private Integer marketShared;

    private Integer agentCount;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}
