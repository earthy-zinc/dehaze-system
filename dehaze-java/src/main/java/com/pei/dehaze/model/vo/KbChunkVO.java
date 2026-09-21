package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/** AI 知识库分块视图（对齐 python KnowledgeChunkVO） */
@Schema(description = "知识库分块视图")
@Data
public class KbChunkVO {

    private Long id;

    private Long documentId;

    @Schema(description = "分块序号(从0开始)")
    private Integer chunkIndex;

    private String content;

    private Integer tokenCount;

    @Schema(description = "分块元数据(来源文档/页码/段落/表格行等)")
    private Object metadata;

    private LocalDateTime createTime;
}
