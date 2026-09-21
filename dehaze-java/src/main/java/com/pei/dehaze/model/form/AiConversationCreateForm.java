package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Size;
import java.util.Map;
import lombok.Data;

/**
 * 创建 AI 会话请求
 *
 * @author dehaze
 */
@Data
public class AiConversationCreateForm {

    @Size(max = 255, message = "会话标题长度不能超过255")
    @Schema(description = "会话标题")
    private String title;

    @Size(max = 64, message = "模型标识长度不能超过64")
    @Schema(description = "会话使用的模型标识")
    private String model;

    @Schema(description = "系统提示词")
    private String systemPrompt;

    @Schema(description = "模型参数配置")
    private Map<String, Object> modelConfig;

    @Schema(description = "绑定的API Key ID")
    private Long apiKeyId;

    @Size(max = 64, message = "Agent编码长度不能超过64")
    @Schema(description = "会话使用的Agent编码(为空使用默认Agent)")
    private String agentCode;

    @Schema(description = "类似问题推荐开关")
    private Boolean suggestionsEnabled;

    @Size(max = 32, message = "会话场景长度不能超过32")
    @Schema(description = "会话场景(general/image_dispatch/multi_step/algorithm_recommend/scheduled_task)")
    private String scene;

}
