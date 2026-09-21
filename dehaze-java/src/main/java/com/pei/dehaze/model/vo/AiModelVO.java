package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Schema(description = "AI 模型视图")
@Data
public class AiModelVO {

    private Long id;

    private Long providerId;

    private String modelId;

    private String modelType;

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

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
