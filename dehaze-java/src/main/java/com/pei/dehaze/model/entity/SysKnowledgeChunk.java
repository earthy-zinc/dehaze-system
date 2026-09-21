package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/** AI 知识库分块（表 sys_knowledge_chunk，只追加，向量存 ES） */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysKnowledgeChunk extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long documentId;

    private Long knowledgeBaseId;

    private Integer chunkIndex;

    private Integer sectionIndex;

    private String sectionPath;

    private String content;

    private Integer tokenCount;

    private String metadata;
}
