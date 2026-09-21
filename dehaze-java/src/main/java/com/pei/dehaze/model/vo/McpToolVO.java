package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.Map;

@Schema(description = "MCP 工具清单项")
@Data
public class McpToolVO {

    private String name;

    private String description;

    private Map<String, Object> inputSchema;
}
