package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Schema(description = "知识库视图")
@Data
public class KbVO {

    private Long id;

    private String name;

    private String description;

    private String visibility;

    private String embeddingProvider;

    private String embeddingModel;

    private String chunkingStrategy;

    private Integer chunkSize;

    private Integer chunkOverlap;

    private String searchStrategy;

    private BigDecimal hybridWeight;

    private Integer topK;

    private BigDecimal scoreThreshold;

    private Integer enableRerank;

    private String rerankModel;

    private Integer documentCount;

    private Integer chunkCount;

    private Long totalTokens;

    private Integer status;

    private Long createBy;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}
