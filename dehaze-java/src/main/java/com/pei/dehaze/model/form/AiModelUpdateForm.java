package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.Map;

/**
 * AI 模型更新表单：仅提交需要变更的字段（null 视为不修改）。
 *
 * <p>modelType/dimension 仅用于承载请求值以触发"创建后不可改"校验（A0502），
 * 不参与落库。
 */
@Schema(description = "AI 模型更新表单")
@Data
public class AiModelUpdateForm {

    private Long providerId;

    private String modelType;

    @Min(1)
    private Long dimension;

    @Size(min = 1, max = 128)
    private String displayName;

    @Min(1)
    private Integer maxContextTokens;

    @Min(1)
    private Integer maxOutputTokens;

    private Boolean supportsMultimodal;

    private Boolean supportsToolCall;

    private Boolean supportsStreaming;

    private Boolean supportsPromptCache;

    private Boolean supportsStructuredOutput;

    private Map<String, Object> extraRequestParams;

    private Long fallbackModelId;

    @Min(0)
    private Integer promptCachePrefixLen;

    private Integer status;

    @Min(0)
    @Max(2)
    private Integer vipLevel;
}
