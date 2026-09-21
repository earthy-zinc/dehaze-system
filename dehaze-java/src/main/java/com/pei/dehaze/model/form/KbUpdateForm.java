package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.math.BigDecimal;

/**
 * 知识库编辑表单（仅可编辑项；embedding_model / chunking_strategy 携带即被服务端拒绝）。
 */
@Schema(description = "知识库编辑表单")
@Data
public class KbUpdateForm {

    @Size(min = 1, max = 255)
    private String name;

    private String description;

    /** python 侧为 Literal["vector","keyword","hybrid"]，非法值在请求校验阶段即拒绝（非空时才校验） */
    @Pattern(regexp = "vector|keyword|hybrid")
    private String searchStrategy;

    @DecimalMin("0")
    @DecimalMax("1")
    private BigDecimal hybridWeight;

    @Min(1)
    @Max(100)
    private Integer topK;

    /** python 侧为 lt=1（左闭右开），上界不可取等 */
    @DecimalMin("0")
    @DecimalMax(value = "1", inclusive = false)
    private BigDecimal scoreThreshold;

    private Boolean enableRerank;

    @Size(max = 64)
    private String rerankModel;

    @Size(max = 64)
    private String embeddingModel;

    /**
     * python 侧为 Literal[fixed|semantic|recursive|qa|table]；本字段创建后不可修改（携带即被服务端拒绝），
     * 但非法字面量在 python 属请求校验阶段（业务码 A0400），故此处对齐校验层——合法值仍由服务层报 A0500"不可修改"。
     */
    @Pattern(regexp = "fixed|semantic|recursive|qa|table")
    private String chunkingStrategy;
}
