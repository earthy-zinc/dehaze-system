package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/** AI 对话 Skill 主表（表 sys_ai_skill，F-M08-006） */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiSkill extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;

    private String description;

    private String scene;

    /** SKILL.md 指令正文（frontmatter 之外的内容） */
    private String instruction;

    private String license;

    private String compatibility;

    /** SKILL.md frontmatter metadata（JSON 文本） */
    private String metadata;

    private String allowedTools;

    /** 0:禁用;1:启用，禁用后不进入执行侧索引 */
    private Integer status;

    /** builtin/admin */
    private String source;

    /** 是否共享至 Skill 市场(0:否;1:是) */
    private Integer marketShared;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}
