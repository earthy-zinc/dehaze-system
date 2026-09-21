package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.util.List;
import java.util.Map;
import lombok.Data;

/**
 * 创建智能体请求
 *
 * @author dehaze
 */
@Data
public class AiAgentCreateForm {

    @NotBlank(message = "Agent编码不能为空")
    @Size(max = 64, message = "Agent编码长度不能超过64")
    @Schema(description = "Agent唯一编码")
    private String agentCode;

    @NotBlank(message = "Agent名称不能为空")
    @Size(max = 128, message = "Agent名称长度不能超过128")
    @Schema(description = "Agent显示名称")
    private String name;

    @Size(max = 512, message = "Agent描述长度不能超过512")
    @Schema(description = "Agent描述")
    private String description;

    @Schema(description = "系统提示词(Markdown)")
    private String systemPrompt;

    @NotBlank(message = "模型标识不能为空")
    @Size(max = 64, message = "模型标识长度不能超过64")
    @Schema(description = "关联模型标识")
    private String modelId;

    @Schema(description = "推理范式(auto/direct/react/plan_execute/reflexion)")
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

    /** python AgentCreate.sort_order 为 ge=0 且无上界，故仅约束下限 */
    @Min(value = 0, message = "排序序号不能小于0")
    @Schema(description = "排序序号")
    private Integer sortOrder;

    @Min(value = 0, message = "状态不能小于0")
    @Max(value = 1, message = "状态不能大于1")
    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;

}
