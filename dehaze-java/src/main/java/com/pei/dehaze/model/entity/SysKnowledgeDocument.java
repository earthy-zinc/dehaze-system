package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/** AI 知识库文档（表 sys_knowledge_document） */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysKnowledgeDocument extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long knowledgeBaseId;

    private Long fileId;

    private String title;

    /** manual/upload/url/algorithm_sync/experience */
    private String source;

    private Integer version;

    /** auto/ocr/text/table */
    private String parsingStrategy;

    private String content;

    private String rawContent;

    private Integer chunkCount;

    private Long totalTokens;

    /** pending/processing/completed/failed */
    private String processingStatus;

    private String error;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}
