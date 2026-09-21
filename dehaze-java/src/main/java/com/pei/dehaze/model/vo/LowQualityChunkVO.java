package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Schema(description = "低质量片段视图（被点踩片段）")
@Data
public class LowQualityChunkVO {

    private Long chunkId;

    private String content;

    private Long documentId;

    private Integer thumbsDownCount;
}
