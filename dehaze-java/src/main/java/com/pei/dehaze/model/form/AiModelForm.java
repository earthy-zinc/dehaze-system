package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.Map;

@Schema(description = "AI 模型创建表单")
@Data
public class AiModelForm {

    @NotNull(message = "所属供应商不能为空")
    private Long providerId;

    @NotBlank(message = "模型标识不能为空")
    @Size(min = 1, max = 64)
    private String modelId;

    @Schema(description = "模型类型(chat:对话;embedding:向量;rerank:重排)")
    private String modelType = "chat";

    @Schema(description = "embedding 向量维度(modelType=embedding 时必填)")
    @Min(1)
    private Long dimension;

    @NotBlank(message = "显示名称不能为空")
    @Size(min = 1, max = 128)
    private String displayName;

    @Min(1)
    private Integer maxContextTokens = 4096;

    @Min(1)
    private Integer maxOutputTokens = 4096;

    private Boolean supportsMultimodal = false;

    private Boolean supportsToolCall = false;

    private Boolean supportsStreaming = true;

    private Boolean supportsPromptCache = false;

    private Boolean supportsStructuredOutput = false;

    @Schema(description = "厂商私有请求参数(如 enable_thinking/reasoning_effort)")
    private Map<String, Object> extraRequestParams;

    @Schema(description = "降级模型主键(关联 sys_ai_model.id)")
    private Long fallbackModelId;

    @Min(0)
    private Integer promptCachePrefixLen = 0;

    private Integer status = 1;

    @Min(0)
    @Max(2)
    private Integer vipLevel = 0;
}
