package com.pei.clarity.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.Map;
import lombok.Data;

@Schema(description = "MCP 工具清单项")
@Data
public class McpToolVO {

    private String name;

    private String description;

    private Map<String, Object> inputSchema;
}
