package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Size;
import java.util.List;
import java.util.Map;
import lombok.Data;

/**
 * 更新智能体请求
 *
 * @author dehaze
 */
@Data
public class AiAgentUpdateForm {

    @Size(max = 128, message = "Agent名称长度不能超过128")
    @Schema(description = "Agent显示名称")
    private String name;

    @Size(max = 512, message = "Agent描述长度不能超过512")
    @Schema(description = "Agent描述")
    private String description;

    @Schema(description = "系统提示词(Markdown)")
    private String systemPrompt;

    @Size(max = 64, message = "模型标识长度不能超过64")
    @Schema(description = "关联模型标识")
    private String modelId;

    @Schema(description = "推理范式")
    private String reasoningMode;

    @Schema(description = "推理参数配置")
    private Map<String, Object> config;

    @Schema(description = "是否可作为子Agent")
    private Boolean isSubagent;

    @Schema(description = "是否为Team团队")
    private Boolean isTeam;

    @Schema(description = "是否对外暴露为A2A子Agent")
    private Boolean isExposed;

    @Schema(description = "文件系统权限规则")
    private List<Map<String, Object>> permissions;

    @Schema(description = "分类标签")
    private List<String> tags;

    /** python AgentUpdate.sort_order 为 None + ge=0（可空、仅下限，勿加 @NotNull/@Max） */
    @Min(value = 0, message = "排序序号不能小于0")
    @Schema(description = "排序序号")
    private Integer sortOrder;

}
