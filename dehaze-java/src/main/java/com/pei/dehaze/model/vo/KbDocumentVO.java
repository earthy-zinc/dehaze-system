package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/** 知识库文档视图（列表接口不返回 content 大字段） */
@Schema(description = "知识库文档视图")
@Data
public class KbDocumentVO {

    private Long id;

    private Long knowledgeBaseId;

    private Long fileId;

    private String title;

    private String source;

    private Integer version;

    private String parsingStrategy;

    private String content;

    private String rawContent;

    private Integer chunkCount;

    private Long totalTokens;

    private String processingStatus;

    private String error;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}
