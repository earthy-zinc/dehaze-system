package com.pei.dehaze.model.form;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;
import lombok.Data;

/**
 * 设置 Agent MCP 命名空间请求
 *
 * @author dehaze
 */
@Data
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class AiAgentMcpForm {

    @Schema(description = "MCP 命名空间列表(覆盖式更新)")
    private List<String> mcpNamespaces;

}
