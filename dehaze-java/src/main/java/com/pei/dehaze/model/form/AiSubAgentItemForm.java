package com.pei.dehaze.model.form;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 子 Agent 关联项
 *
 * @author dehaze
 */
@Data
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class AiSubAgentItemForm {

    @NotNull(message = "子Agent ID不能为空")
    @Schema(description = "子Agent ID")
    private Long agentId;

    @Schema(description = "外部A2A端点ID(NULL为本地子Agent)")
    private Long endpointId;

    @Schema(description = "优先级(数字越小越优先)")
    private Integer priority;

}
