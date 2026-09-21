package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/** 分块片段反馈（表 sys_knowledge_chunk_feedback，rating=-1 即低质量片段） */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysKnowledgeChunkFeedback extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long chunkId;

    private Long userId;

    /** 1:点赞;-1:点踩 */
    private Integer rating;

    private String comment;
}
