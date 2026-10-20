package com.pei.clarity.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import lombok.Data;

@Schema(description = "AI 模型视图")
@Data
public class AiModelVO {

    private Long id;

    private Long providerId;

    private String modelId;

    private String modelType;

    @Schema(description = "任务归类标签")
    private List<String> taskTags;

    private Long dimension;

    private String displayName;

    private Integer maxContextTokens;

    private Integer maxOutputTokens;

    private Integer supportsMultimodal;

    private Integer supportsToolCall;

    private Integer supportsStreaming;

    private Integer supportsPromptCache;

    private Integer supportsStructuredOutput;

    private Map<String, Object> extraRequestParams;

    private Long fallbackModelId;

    private Integer promptCachePrefixLen;

    private Integer status;

    private Integer vipLevel;

    private Integer lastTestStatus;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime lastTestAt;

    private String lastTestError;

    @Schema(description = "近24h真实调用次数")
    private Integer calls24h;

    @Schema(description = "近24h调用成功率(百分比)，无调用为 null")
    private Integer successRate24h;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime lastCallAt;

    @Schema(description = "速度档位(fast/medium/slow/unknown)")
    private String speedTier;

    @Schema(description = "是否作为其他启用模型的降级目标")
    private Boolean isFallbackTarget;

    @Schema(description = "是否其模型类型的平台默认模型(字典 ai_model.default_model[.{modelType}])；仅启用模型列表返回")
    private Boolean isDefault;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
