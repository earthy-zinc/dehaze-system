package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Size;
import java.util.Map;
import lombok.Data;

/**
 * 更新 AI 会话请求
 *
 * <p>pinned/status 不加数值约束：python {@code ConversationUpdate} 这两字段是无界裸 int。
 *
 * @author dehaze
 */
@Data
public class AiConversationUpdateForm {

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

    @Schema(description = "是否置顶(0:否;1:是)")
    private Integer pinned;

    @Schema(description = "会话状态(1:活跃;2:已归档)")
    private Integer status;

    @Size(max = 64, message = "Agent编码长度不能超过64")
    @Schema(description = "切换Agent编码(下一条消息生效)")
    private String agentCode;

    @Schema(description = "类似问题推荐开关")
    private Boolean suggestionsEnabled;

}
