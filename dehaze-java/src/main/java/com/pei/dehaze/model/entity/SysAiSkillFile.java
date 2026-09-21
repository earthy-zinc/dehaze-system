package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * SKILL 目录文件清单（表 sys_ai_skill_file）。
 *
 * <p>文件内容存对象存储（对象 key = skills/{skill_id}/{path}），本表仅存清单。
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiSkillFile extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long skillId;

    private String path;

    private Long fileSize;

    private String fileType;
}
